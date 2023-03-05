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
from common.dataset_utils import FaceDataset
from common.ops import convert_to_ddp
from common.metrics import metric_computation
from common.utils import init_weights, vector_difference, pcc_ccc_loss
from common.big_model import MlpMixer, CONFIGS

import cvxpy as cp
#from common.cvx_utils import OptLayer
from common.sparsemax import Sparsemax

import wandb
from fabulous.color import fg256


class AVCE(BasicTask):

    def set_loader(self):
        opt = self.opt

        training_path = opt.data_path+'training.csv'
        validation_path = opt.data_path+'validation.csv'

        face_dataset = FaceDataset(csv_file=training_path,
                                   root_dir=opt.data_path,
                                   transform=transforms.Compose([
                                       transforms.Resize(256), transforms.RandomCrop(size=224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                   ]), inFolder=None)
    
        face_dataset_val = FaceDataset(csv_file=validation_path,
                                       root_dir=opt.data_path,
                                       transform=transforms.Compose([
                                           transforms.Resize(256), transforms.CenterCrop(size=224),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                       ]), inFolder=None)
    
    
        train_loader = DataLoader(face_dataset, batch_size=opt.tr_batch_size, shuffle=True)
        val_loader   = DataLoader(face_dataset_val, batch_size=opt.te_batch_size, shuffle=False)  # False

        self.train_loader = train_loader
        self.val_loader = val_loader


    def set_model(self):
        opt = self.opt

        from common.avce_networks import encoder_Alex, encoder_R18, regressor_Alex, regressor_R18, spregressor, vregressor
        if opt.model == 'alexnet':
            print(fg256("yellow", '[network] AlexNet loaded.'))
            encoder      = encoder_Alex().cuda()
            regressor    = regressor_Alex().cuda()
            sp_regressor = spregressor().cuda()
            v_regressor  = vregressor().cuda()
        elif opt.model == 'resnet18':
            print(fg256("yellow", '[network] ResNet18 loaded.'))
            encoder    = encoder_R18().cuda()
            regressor  = regressor_R18().cuda()
            sp_regressor = spregressor().cuda()
            v_regressor  = vregressor().cuda()

        e_opt = torch.optim.Adam(encoder.parameters(),     lr = 5e-5, betas = (0.5, 0.99))
        r_opt = torch.optim.Adam(regressor.parameters(),   lr = 5e-5, betas = (0.5, 0.99))
        s_opt = torch.optim.SGD(sp_regressor.parameters(), lr = 1e-2, momentum=0.9)
        v_opt = torch.optim.SGD(v_regressor.parameters(),  lr = 1e-2, momentum=0.9)

        self.encoder      = encoder
        self.regressor    = regressor
        self.sp_regressor = sp_regressor
        self.v_regressor  = v_regressor

        self.e_opt = e_opt
        self.r_opt = r_opt
        self.s_opt = s_opt
        self.v_opt = v_opt

        self.e_lr_scheduler = lr_scheduler.MultiStepLR(self.e_opt, milestones=[5e2,10e2,20e2,30e2,50e2], gamma=0.8)
        self.r_lr_scheduler = lr_scheduler.MultiStepLR(self.r_opt, milestones=[5e2,10e2,20e2,30e2,50e2], gamma=0.8)
        self.s_lr_scheduler = lr_scheduler.MultiStepLR(self.s_opt, milestones=[5e2,10e2,20e2,30e2,50e2], gamma=0.8)
        self.v_lr_scheduler = lr_scheduler.MultiStepLR(self.v_opt, milestones=[5e2,10e2,20e2,30e2,50e2], gamma=0.8)

    def validate(self, current_info, n_iter):
        opt = self.opt
        MSE = nn.MSELoss()
        cnt = 0

        current_dir, current_time = current_info[0], current_info[1]
        self.encoder.eval()
        self.regressor.eval()
        self.sp_regressor.eval()
        self.v_regressor.eval()

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

        radian, amp_factor = 57.2958, 1.0
        cnt, print_check, margin = 0, 50, 0.5

        ALPHA = Variable(torch.FloatTensor([0.5]).cuda(), requires_grad=True)
        BETA  = Variable(torch.FloatTensor([0.005]).cuda(), requires_grad=True)
        GAMMA = Variable(torch.FloatTensor([0.5]).cuda(), requires_grad=True)

        self.encoder.train()
        self.regressor.train()
        self.sp_regressor.train()
        self.v_regressor.train()

        SP = Sparsemax(dim=1)
        SM = torch.nn.Softmax(dim=1)

#        # SparseMax
#        z = cp.Variable(32)
#        x = cp.Parameter(32)
#        
#        f_ = lambda z,x : cp.sum_squares(z - x) if isinstance(z, cp.Variable) else torch.sum((x-z)**2)
#        g_ = lambda z,x : -z
#        h_ = lambda z,x: cp.sum(z) - 1 if isinstance(z, cp.Variable) else z.sum() - 1
#        sp_layer = OptLayer([z], [x], f_, [g_], [h_])
#        
#        # SoftMax
#        zs = cp.Variable(32)
#        xs = cp.Parameter(32)
#        
#        fs_ = lambda z,x: -z@x - cp.sum(cp.entr(z)) if isinstance(z, cp.Variable) else -z@x + z@torch.log(z)
#        hs_ = lambda z,x: cp.sum(z) - 1 if isinstance(z, cp.Variable) else z.sum() - 1
#        sm_layer = OptLayer([zs], [xs], fs_, [], [hs_])

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
        
                inputs, correct_labels = Variable(data.cuda()), Variable(emotions.cuda())

                self.e_opt.zero_grad()
                self.r_opt.zero_grad()
                self.s_opt.zero_grad()
                self.v_opt.zero_grad()

                # ---------------
                # Train regressor
                # ---------------
                z = self.encoder(inputs)
                scores, z_btl = self.regressor(z)  # [2], [32]
    
                z_sp_btl = SP(z_btl)
                z_sm_btl = SM(z_btl)
                sp_scores = self.sp_regressor(z_sp_btl)
                sm_scores = self.sp_regressor(z_sm_btl)
    
                sp_scores_norm = torch.norm(sp_scores, p=2, dim=1)
                sm_scores_norm = torch.norm(sm_scores, p=2, dim=1)
                scores_norm = torch.norm(scores, p=2, dim=1)
                diff_norm_pos = amp_factor * torch.abs(sp_scores_norm - scores_norm)
                diff_norm_neg = amp_factor * torch.abs(sm_scores_norm - scores_norm)

                inner_product = (sp_scores * scores).sum(dim=1)
                a_norm = sp_scores.pow(2).sum(dim=1).pow(0.5) + 1e-8
                b_norm = scores.pow(2).sum(dim=1).pow(0.5) + 1e-8
                cos_sp = inner_product / (2 * a_norm * b_norm)
                angle_sp = torch.acos(cos_sp)
                with torch.no_grad():
                    angle_sp_mean = angle_sp.mean()
                    angle_sp_std = torch.std(angle_sp)
    
                inner_product = (sm_scores * scores).sum(dim=1)
                a_norm = sm_scores.pow(2).sum(dim=1).pow(0.5) + 1e-8
                b_norm = scores.pow(2).sum(dim=1).pow(0.5) + 1e-8
                cos_sm = inner_product / (2 * a_norm * b_norm)
                angle_sm = torch.acos(cos_sm)
                with torch.no_grad():
                    angle_sm_mean = angle_sm.mean()
                    angle_sm_std = torch.std(angle_sm)

                rpc_loss = (angle_sp+diff_norm_pos).mean() - ALPHA * (angle_sm+diff_norm_neg).mean() \
                        - 0.5 * BETA * (angle_sp+diff_norm_pos).pow(2).mean() \
                        - 0.5 * GAMMA * (angle_sm+diff_norm_neg).pow(2).mean()

                pcc_loss, ccc_loss, ccc_v, ccc_a = pcc_ccc_loss(correct_labels, scores)
    
                MSE_v = MSE(scores[:,0], correct_labels[:,0])
                MSE_a = MSE(scores[:,1], correct_labels[:,1])
    
                self.e_opt.zero_grad()
                self.r_opt.zero_grad()
                self.s_opt.zero_grad()
                loss = (MSE_v + MSE_a) - 1e-3 * rpc_loss.cuda() + (0.5 * pcc_loss + 0.5 * ccc_loss)
                loss.backward(retain_graph=True)
    
                self.e_opt.step()
                self.r_opt.step()
                self.s_opt.step()


                # ---------------------------
                # Metric-based Regularization
                # ---------------------------
                self.e_opt.zero_grad()
                self.v_opt.zero_grad()

                z = self.encoder(inputs)
                _, z_btl = self.regressor(z)
                z_sp_btl = SP(z_btl)
                z_sm_btl = SM(z_btl)

                v_btl = self.v_regressor(z_btl)
                v_sp_btl = self.v_regressor(z_sp_btl)
                v_sm_btl = self.v_regressor(z_sm_btl)

                holding_vector1 = torch.norm(vector_difference(v_sp_btl,v_btl), p=2, dim=1, keepdim=True)
                holding_vector2 = torch.norm(vector_difference(v_sm_btl,v_btl), p=2, dim=1, keepdim=True)
                one_vector = torch.ones_like(holding_vector1)
    
                only_pushing_loss = torch.mean(F.relu(1.0 - torch.norm(v_sp_btl - v_sm_btl, p=2, dim=1).pow(2)))
                reg_loss = only_pushing_loss.cuda() + 0.01 * (torch.mean(holding_vector1 - one_vector) + torch.mean(holding_vector2 - one_vector)).cuda()  # ph
                reg_loss.backward()

                self.e_opt.step()
                self.v_opt.step()

                if opt.online_tracker:
                    wandb.log({
                        "loss": loss.item(),
                        'RPC': rpc_loss.item(),
                        "Enc_lr": aa, "Reg_lr": bb,
                        "epoch": epoch, "ccc_v": ccc_v.item(), "ccc_a": ccc_a.item(),
                        "MSE (v)": MSE_v, "MSE (a)": MSE_a
                    })
   
                if n_iter % opt.print_check == 0 and n_iter > 0:
                    torch.save(self.encoder.state_dict(),   opt.save_path+'encoder_{}_{}.t7'.format(n_iter, epoch))
                    torch.save(self.regressor.state_dict(), opt.save_path+'regressor_{}_{}.t7'.format(n_iter, epoch))
                    torch.save(self.sp_regressor.state_dict(), opt.save_path+'sp_regressor_{}_{}.t7'.format(n_iter, epoch))
                    self.validate(current_info, n_iter)

                n_iter = n_iter + 1

                self.e_lr_scheduler.step()
                self.r_lr_scheduler.step()
                self.s_lr_scheduler.step()
                self.v_lr_scheduler.step()
