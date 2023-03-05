import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F
import torch.cuda.amp as amp
import numpy as np
from tqdm import tqdm

from . import BasicTask
from common.data_prefetcher import DataPrefetcher
from common.dataset_utils import FaceDataset
from common.ops import convert_to_ddp
from common.metrics import metric_computation
from common.utils import init_weights, vector_difference, pcc_ccc_loss
from common.big_model import MlpMixer, CONFIGS
from common.caf_utils import estimate_ratio_compute_mmd, dist_l2, cumulative_thresholding, mutual_info
from common.aknn_alg import aknn, predict_nn_rule, calc_nbrs_exact, knn_rule

import wandb

''' About `kmeans`
Custom install package
@link:
    https://github.com/subhadarship/kmeans_pytorch
'''
#from kmeans_pytorch import kmeans
from fabulous.color import fg256


class CAF(BasicTask):

    def set_loader(self):
        opt = self.opt

        training_path = opt.data_path+'train/training_x2_high_res.csv'
        validation_path = opt.data_path+'val/zsample_x10/zvalidation_x10_final.csv'

        face_dataset = FaceDataset(csv_file=training_path,
                                   root_dir=opt.data_path+'train/',
                                   transform=transforms.Compose([
                                       transforms.Resize(256), transforms.RandomCrop(size=224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                   ]), inFolder=None)
    
        face_dataset_val = FaceDataset(csv_file=validation_path,
                                       root_dir=opt.data_path+'val/',
                                       transform=transforms.Compose([
                                           transforms.Resize(256), transforms.CenterCrop(size=224),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                       ]), inFolder=None)
    
    
        train_loader = DataLoader(face_dataset, batch_size=opt.tr_batch_size, shuffle=True)
        val_loader   = DataLoader(face_dataset_val, batch_size=opt.te_batch_size, shuffle=False)  # False

        #self.prefetcher = DataPrefetcher(train_loader)
        self.train_loader = train_loader
        self.val_loader = val_loader


    def set_model(self):
        opt = self.opt

        from common.caf_networks import encoder_Alex, encoder_R18, regressor_Alex, regressor_R18, discriminator, mine
        if opt.model == 'alexnet':
            print(fg256("yellow", '[network] AlexNet loaded.'))
            encoder   = encoder_Alex().cuda()
            regressor = regressor_Alex().cuda()
            disc      = discriminator().cuda()
            mine      = mine().cuda()
        elif opt.model == 'resnet18':
            print(fg256("yellow", '[network] ResNet18 loaded.'))
            encoder   = encoder_R18().cuda()
            regressor = regressor_R18().cuda()
            disc      = discriminator().cuda()
            mine      = mine().cuda()

        e_opt = torch.optim.Adam(encoder.parameters(),     lr = 1e-4, betas = (0.5, 0.9))
        r_opt = torch.optim.Adam(regressor.parameters(),   lr = 1e-4, betas = (0.5, 0.9))
        d_opt = torch.optim.Adam(disc.parameters(),        lr = 1e-4, betas = (0.5, 0.9))
        m_opt = torch.optim.Adam(mine.parameters(),        lr = 1e-4, betas = (0.5, 0.9))

        self.encoder   = encoder
        self.regressor = regressor
        self.disc      = disc
        self.mine      = mine

        self.e_opt = e_opt
        self.r_opt = r_opt
        self.d_opt = d_opt
        self.m_opt = m_opt

        self.e_lr_scheduler = lr_scheduler.MultiStepLR(self.e_opt, milestones=[5e2,10e2,20e2,30e2,50e2], gamma=0.8)
        self.r_lr_scheduler = lr_scheduler.MultiStepLR(self.r_opt, milestones=[5e2,10e2,20e2,30e2,50e2], gamma=0.8)
        self.d_lr_scheduler = lr_scheduler.MultiStepLR(self.d_opt, milestones=[5e2,10e2,20e2,30e2,50e2], gamma=0.8)
        self.m_lr_scheduler = lr_scheduler.MultiStepLR(self.m_opt, milestones=[5e2,10e2,20e2,30e2,50e2], gamma=0.8)

    def validate(self, current_info, n_iter):
        opt = self.opt
        MSE = nn.MSELoss()
        cnt = 0

        current_dir, current_time = current_info[0], current_info[1]
        self.encoder.eval()
        self.regressor.eval()
        self.disc.eval()
        self.mine.eval()

        rmse_v, rmse_a = 0., 0.
        prmse_v, prmse_a = 0., 0.
        inputs_list, all_z_list, scores_list, labels_list = [], [], [], []
        with torch.no_grad():
            for _, data_i in enumerate(self.val_loader):
    
                data, emotions = data_i['image'], data_i['va']
                valence = np.expand_dims(np.asarray(emotions[0]), axis=1)  # [64, 1]
                arousal = np.expand_dims(np.asarray(emotions[1]), axis=1)
                emo_np = np.concatenate([valence, arousal], axis=1)
                emotions = torch.from_numpy(emo_np).float()
    
                inputs, correct_labels = Variable(data.cuda()), Variable(emotions.cuda())
                scores, all_z = self.regressor(self.encoder(inputs))

                inputs_list.append(inputs.detach().cpu().numpy())
                all_z_list.append(all_z.detach().cpu().numpy())
                scores_list.append(scores.detach().cpu().numpy())
                labels_list.append(correct_labels.detach().cpu().numpy())
    
                RMSE_valence = MSE(scores[:,0], correct_labels[:,0])**0.5
                RMSE_arousal = MSE(scores[:,1], correct_labels[:,1])**0.5
    
                rmse_v += RMSE_valence.item(); rmse_a += RMSE_arousal.item()
                cnt = cnt + 1

        if opt.save_npy:
            if n_iter % opt.print_check == 0:
                print(fg256("cyan", '\n[INFO] Images and features for EVAL save.'))
                all_z_th  = np.concatenate(all_z_list)   # shape: [1940, 512]
                scores_th = np.concatenate(scores_list)
                labels_th = np.concatenate(labels_list)
    
                np.save(opt.save_path+'eval_all_z_{}.npy'.format(n_iter), all_z_th)
                np.save(opt.save_path+'eval_scores_{}.npy'.format(n_iter), scores_th)
                np.save(opt.save_path+'eval_labels_{}.npy'.format(n_iter), labels_th)
    
                if n_iter == opt.print_check:
                    inputs_th = np.concatenate(inputs_list)
                    np.save(opt.save_path+'eval_inputs.npy', inputs_th)


        PCC_v, PCC_a, CCC_v, CCC_a, SAGR_v, SAGR_a, final_rmse_v, final_rmse_a = \
                metric_computation([rmse_v,rmse_a], scores_list, labels_list, cnt)
    
        # write results to log file
        if n_iter == opt.print_check:
            with open(current_dir+'/log/'+current_time+'.txt', 'a') as f:
                f.writelines(['{}\n\n'.format(opt)])

        with open(current_dir+'/log/'+current_time+'.txt', 'a') as f:
            f.writelines(['Itr:  \t {}, \
                    \nPCC:  \t{}|\t {}, \
                    \nCCC:  \t{}|\t {}, \
                    \nSAGR: \t{}|\t {}, \
                    \nRMSE: \t{}|\t {}\n\n'
                .format(
                    n_iter,
                    PCC_v[0,1],   PCC_a[0,1],
                    CCC_v[0,1],   CCC_a[0,1],
                    SAGR_v,       SAGR_a,
                    final_rmse_v, final_rmse_a,
            )])


    def train(self, current_info):
        opt = self.opt
        if opt.online_tracker:
            wandb.init(project=opt.project_title)

        MSE = nn.MSELoss()
        n_iter = 0
        n_cluster = 3
        rho = 1e-6

        one = torch.FloatTensor([1])
        mone = (one * -1).cuda()
        Lambda = torch.FloatTensor([0]).cuda()
        Lambda = Variable(Lambda, requires_grad=True)

        cnt, print_check, margin = 0, 50, 0.5

        self.encoder.train()
        self.regressor.train()
        self.disc.train()
        self.mine.train()


        for epoch in range(opt.num_epoch):
            print('epoch ' + str(epoch) + '/' + str(opt.num_epoch-1))

            epoch_iterator = tqdm(self.train_loader,
                                  desc="Training (X / X Steps) (loss=X.X)",
                                  bar_format="{l_bar}{r_bar}",
                                  dynamic_ncols=True)
            for _, data_i in enumerate(epoch_iterator):

                for enc_param_group in self.e_opt.param_groups:
                    aa = enc_param_group['lr']
                for reg_param_group in self.r_opt.param_groups:
                    bb = reg_param_group['lr']

                data, emotions = data_i['image'], data_i['va']
        
                valence = np.expand_dims(np.asarray(emotions[0]), axis=1)  # [64, 1]
                arousal = np.expand_dims(np.asarray(emotions[1]), axis=1)
                emo_np = np.concatenate([valence, arousal], axis=1)
                emotions = torch.from_numpy(emo_np).float()

                spec_loss = 0
                hard_idx, weak_idx = [], []


#                # 1) Kmeans-based thresholding
#                _, ctr = kmeans(
#                        X=emotions.type(torch.FloatTensor),
#                        num_clusters=n_cluster, distance='euclidean', device=torch.device('cpu')
#                )
#    
#                if torch.sum(ctr != ctr):  # To check whether is nan value or not
#                    th = 1.3
#                else:
#                    th_v = torch.mean(ctr[:,0], dim=0, k
#                    th_a = torch.mean(ctr[:,1], dim=0, keepdim=True)
#                    th = th_v.pow(2) + th_a.pow(2)
#                    print(fg256("cyan", 'th_v | _a | _mean value is : {} | {} | {}'.format(th_v, th_a, th)))

                # 2) Histogram-based thresholding
                th_v, th_a = cumulative_thresholding(valence, 0.75, 0.95), cumulative_thresholding(arousal, 0.75, 0.95)
                th = th_v.pow(2) + th_a.pow(2)
                offset = 0.1  # 0.25

                for i in range(emotions.size()[0]):
                    if emotions[i][0].pow(2) + emotions[i][1].pow(2) > th:  # > 15.0:
                        hard_idx.append(i)
                    elif emotions[i][0].pow(2) + emotions[i][1].pow(2) < th:  # < 10.0:
                        weak_idx.append(i)
    
                if len(hard_idx) == 0 or len(weak_idx) == 0:
                    print(fg256("red", "BBB!!"))
                    for i in range(emotions.size()[0]):
                        if emotions[i][0].pow(2) + emotions[i][1].pow(2) > 0.65:
                            hard_idx.append(i)
                        elif emotions[i][0].pow(2) + emotions[i][1].pow(2) < 0.3:
                            weak_idx.append(i)
    
                if len(hard_idx) == 0: hard_idx = weak_idx
    
                hard_idx = hard_idx[:min(len(weak_idx), len(hard_idx))]
                weak_idx = weak_idx[:min(len(weak_idx), len(hard_idx))]
                hard_idx = torch.tensor([hard_idx])
                weak_idx = torch.tensor([weak_idx])

        
                inputs, correct_labels = Variable(data.cuda()), Variable(emotions.cuda())


                # ---
                # Supervision learning
                # ---
                z = self.encoder(inputs)
                scores = self.regressor(z)

                #PCC(/CCC) losses; original paper (i.e., AAAI2021) didn't include this part!
                pcc_loss, ccc_loss, ccc_v, ccc_a = pcc_ccc_loss(correct_labels, scores)
                loss = MSE(scores, correct_labels) + 0.5 * (pcc_loss + ccc_loss)

                diff_v = scores[:,0] - correct_labels[:,0]
                diff_a = scores[:,1] - correct_labels[:,1]
                RMSE_valence = torch.sqrt(diff_v.pow(2)/scores[:,0].size(0)).mean()
                RMSE_arousal = torch.sqrt(diff_a.pow(2)/scores[:,1].size(0)).mean()

                self.e_opt.zero_grad()
                self.r_opt.zero_grad()

                loss.backward(retain_graph=True)
                self.e_opt.step()
                self.r_opt.step()

                # ---
                # Train discriminator
                # ---
                for p in self.disc.parameters(): p.requires_grad = True

                msize = hard_idx.size(1)//2

                z_high_emo = self.encoder(inputs[hard_idx].squeeze(0))
                z_low_emo  = self.encoder(inputs[weak_idx].squeeze(0))

                if z_high_emo.size(0) == 1 or z_low_emo.size(0) == 1:
                    D_hh_emo = self.disc(z_high_emo)
                    D_ll_emo = self.disc(z_low_emo)
                else:
                    D_hh_emo = self.disc(z_high_emo[:msize*2])
                    D_ll_emo = self.disc(z_low_emo[:msize*2])

                if torch.sum(D_hh_emo != D_hh_emo):  # To check whether is nan value or not
                    print(fg256("red", "`NaN` value occured!"))
                    D_hh_emo = torch.ones_like(D_hh_emo)

                if torch.sum(D_ll_emo != D_ll_emo):  # To check whether is nan value or not
                    print(fg256("red", "`NaN` value occured!"))
                    D_ll_emo = torch.ones_like(D_ll_emo)


#                # KDH-GAN
#                ratio, mmd = estimate_ratio_compute_mmd(
#                        D_ll_emo, D_hh_emo, list([1, 10, 100, 1000])  # []
#                )
#                wandb.log({"Lambda (art. sgd)": Lambda.cpu()})

                mi = mutual_info(D_hh_emo, D_ll_emo)
                mi = torch.clamp(mi, 0.0, 2.5)

                # ---
                # Adaptive lower bound for choosing suitable `k` in balls
                # ---
                nmn = torch.cat([D_hh_emo, D_ll_emo], dim=0).cpu().detach().numpy()
                nbrs_list = calc_nbrs_exact(nmn, k=10)
                confidence = 1.2 / np.sqrt(np.arange(nbrs_list.shape[1])+1)  # 0.75

                adaptive_ks = []
                labels = np.concatenate([np.ones(D_hh_emo.size(0)), np.zeros(D_ll_emo.size(0))], axis=0)
                for i in range(nbrs_list.shape[0]):
                    (_, adaptive_k_ndx, _) = aknn(nbrs_list[i,:], labels, confidence)
                    adaptive_ks.append(adaptive_k_ndx + 1)
                adaptive_ks = np.array(adaptive_ks)
                print(fg256("magenta", "adaptive_ks is ", adaptive_ks))
                print(fg256("cyan", "spectrum of conf. is ", confidence))

                # incremental constant value
#                kappa = 0.1 + epoch * 0.005

                d_ap, d_an = [], []
                minfo = []
                for i in range(0, msize):
                    mi = torch.clamp(mutual_info(D_hh_emo[i], D_ll_emo[-i-1]), 0.0, 2.5)
                    pos_dist = F.relu( ((D_hh_emo[i]-D_hh_emo[-i-1]).pow(2)).sum() )
                    neg_dist = F.relu( mi + confidence[adaptive_ks[i]-2] - ((D_hh_emo[i]-D_ll_emo[-i-1]).pow(2)).sum() )
                    d_ap.append(pos_dist); d_an.append(neg_dist); minfo.append(mi)
                for i in range(msize,D_hh_emo.size(0)):
                    mi = torch.clamp(mutual_info(D_hh_emo[i], D_ll_emo[-i-1]), 0.0, 2.5)
                    pos_dist = F.relu( ((D_hh_emo[i]-D_hh_emo[-i-1]).pow(2)).sum() )
                    neg_dist = F.relu( mi + confidence[adaptive_ks[i]-2] - ((D_hh_emo[i]-D_ll_emo[-i-1]).pow(2)).sum() )
                    d_ap.append(pos_dist); d_an.append(neg_dist); minfo.append(mi)

#                for i in range(0, msize):
#                    mi = torch.clamp(mutual_info(D_hh_emo[i], D_ll_emo[-i-1]), 0.0, 2.5)
#                    pos_dist = F.relu( ((D_hh_emo[i]-D_hh_emo[-i-1]).pow(2)).sum() )
#                    neg_dist = F.relu( mi + kappa - ((D_hh_emo[i]-D_ll_emo[-i-1]).pow(2)).sum() )
#                    d_ap.append(pos_dist); d_an.append(neg_dist); minfo.append(mi)
#                for i in range(msize,D_hh_emo.size(0)):
#                    mi = torch.clamp(mutual_info(D_hh_emo[i], D_ll_emo[-i-1]), 0.0, 2.5)
#                    pos_dist = F.relu( ((D_hh_emo[i]-D_hh_emo[-i-1]).pow(2)).sum() )
#                    neg_dist = F.relu( mi + kappa - ((D_hh_emo[i]-D_ll_emo[-i-1]).pow(2)).sum() )
#                    d_ap.append(pos_dist); d_an.append(neg_dist); minfo.append(mi)
    
                d_ap, d_an = torch.stack(d_ap), torch.stack(d_an)
                minfo = torch.mean(torch.stack(minfo))

                if torch.sum(d_ap != d_ap):
                    d_ap = D_hh_emo; d_an = D_ll_emo

                E_P_f,  E_Q_f  = d_ap.mean(), d_an.mean()
                E_P_f2, E_Q_f2 = (d_ap**2).mean(), (d_an**2).mean()
                constraint = (1 - (0.5*E_P_f2 + 0.5*E_Q_f2))
    
                self.d_opt.zero_grad()
                D_loss = E_P_f + E_Q_f + Lambda * constraint - rho/2 * constraint**2
                D_loss.backward()
                self.d_opt.step()

                # artisnal sgd. We minimize Lambda so Lambda <- Lambda + lr * (-grad)
                Lambda.data += rho * Lambda.grad.data
                Lambda.grad.data.zero_()

                # ---
                # Train encoder (adversarial)
                # ---
                for p in self.disc.parameters(): p.requires_grad = False
    
                fake_z = self.encoder(inputs[weak_idx].squeeze(0))
                fake_emo = self.disc(fake_z)

                self.e_opt.zero_grad()
                fake_loss = -torch.mean(fake_emo) * 0.1
                fake_loss.backward()
                self.e_opt.step()
    
                del hard_idx, weak_idx


                if opt.online_tracker:
                    wandb.log({
                        "loss": loss.item(), "D_loss": D_loss.cpu(), "G_loss": fake_loss.cpu(),
                        "Enc_lr": aa, "Reg_lr": bb,
                        "d_ap": d_ap.cpu(), "d_an": d_an.cpu(), "threshold": th,
                        "Lambda (art. sgd)": Lambda.cpu(),
                        "Confidence": confidence.cpu(), "Mutual information": minfo.cpu(),
                        "epoch": epoch, "ccc_v": ccc_v.item(), "ccc_a": ccc_a.item(),
                        "MSE (v)": MSE_valence, "MSE (a)": MSE_arousal

                    })
   
                if n_iter % opt.print_check == 0 and n_iter > 0:
                    torch.save(self.encoder.state_dict(),   opt.save_path+'encoder_{}_{}.t7'.format(n_iter, epoch))
                    torch.save(self.regressor.state_dict(), opt.save_path+'regressor_{}_{}.t7'.format(n_iter, epoch))
                    torch.save(self.disc.state_dict(), opt.save_path+'disc_{}_{}.t7'.format(n_iter, epoch))
                    self.validate(current_info, n_iter)

                n_iter = n_iter + 1

                self.e_lr_scheduler.step()
                self.r_lr_scheduler.step()
                self.d_lr_scheduler.step()
