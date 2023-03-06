import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F
import torch.cuda.amp as amp
import numpy as np
import copy
from tqdm import tqdm

from . import BasicTask
from common.data_prefetcher import DataPrefetcher
from common.dataset_utils import FaceDataset, AddGaussianNoise
from common.ops import convert_to_ddp
from common.metrics import metric_computation
from common.utils import init_weights, pcc_ccc_loss, encoder_fun, reparameterize, CLIP_Encoding
from common.big_model import MlpMixer, CONFIGS
from common.attention_module import CrossHeadAttention
from common.attackers import Fast_gradient_sign_method, Projected_gradient_descent, PyramidAT, \
                                CAM_FGSM, CAM_PGD, CAM_PyramidAT, \
                                FAAT
from common.purifiers import Purification_basic

import wandb
# from fabulous.color import fg256


class Baseline(BasicTask):

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
                                           transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
                                       ]), inFolder=None)
    
    
        train_loader = DataLoader(face_dataset, batch_size=opt.tr_batch_size, shuffle=True)
        val_loader   = DataLoader(face_dataset_val, batch_size=opt.te_batch_size, shuffle=False)  # False

        self.train_loader = train_loader
        self.val_loader = val_loader


    def set_model(self):
        opt = self.opt

        NGPU = torch.cuda.device_count()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        from common.networks import Encoder_Alex, Encoder_R18, Encoder_SR50, Regressor_Alex, Regressor_R18, Regressor_SR50, Regressor_MMx, Linear_Prob, Linear_Prob_VA, Face_generator_upsample, Neural_decomp_mlp
        if opt.model == 'alexnet':
            print('[network] AlexNet loaded.')
            encoder     = Encoder_Alex().cuda()
            regressor   = Regressor_Alex(opt.latent_dim).cuda()
        elif opt.model == 'resnet18':
            print('[network] ResNet18 loaded.')
            encoder     = Encoder_R18().cuda()
            regressor   = Regressor_R18(opt.latent_dim).cuda()
        elif opt.model == 'mlpmixer':
            print('[network] ViT-style MlpMixer loaded.')
            config      = CONFIGS['Mixer-B_16']
            encoder     = MlpMixer(config, img_size=224, num_classes=2, patch_size=16, latent_dim=opt.latent_dim, zero_head=True)
            encoder.load_from(np.load(opt.vit_path))
            encoder = nn.DataParallel(encoder, device_ids=list(range(NGPU)))  #.to(device)
            regressor = Regressor_MMx(opt.latent_dim).to(device)
    
        e_opt = torch.optim.Adam(encoder.parameters(), lr = opt.e_lr, betas = (0.5, 0.99))
        r_opt = torch.optim.Adam(regressor.parameters(), lr = opt.r_lr, betas = (0.5, 0.99))

        scaler = amp.GradScaler()
        self.scaler = scaler

        self.encoder = encoder
        self.regressor = regressor

        self.e_opt = e_opt
        self.r_opt = r_opt

        self.e_lr_scheduler = lr_scheduler.MultiStepLR(self.e_opt, milestones=[5e2,1e3,2e3,5e3,10e3], gamma=0.8)
        self.r_lr_scheduler = lr_scheduler.MultiStepLR(self.r_opt, milestones=[5e2,1e3,2e3,5e3,10e3], gamma=0.8)


    def validate(self, current_info, n_iter):
        opt = self.opt
        MSE = nn.MSELoss()
        cnt = 0

        current_dir, current_time = current_info[0], current_info[1]
        self.encoder.eval()
        self.regressor.eval()

        rmse_v, rmse_a = 0., 0.
        prmse_v, prmse_a = 0., 0.
        inputs_list, all_z_list, scores_list, labels_list = [], [], [], []
        with torch.no_grad():
            for _, data_i in enumerate(self.val_loader):
    
                data, emotions = data_i['image'], data_i['va']
                valence = np.expand_dims(np.asarray(emotions[0]), axis=1)
                arousal = np.expand_dims(np.asarray(emotions[1]), axis=1)
                emo_np = np.concatenate([valence, arousal], axis=1)
                emotions = torch.from_numpy(emo_np).float()
    
                inputs, correct_labels = Variable(data.cuda()), Variable(emotions.cuda())

                if opt.model == 'mlpmixer':
                    _, z = self.encoder(inputs)
                    scores, all_z = self.regressor(z)
                else:
                    z = self.encoder(inputs)
                    scores, all_z = self.regressor(z)

                inputs_list.append(inputs.detach().cpu().numpy())
                all_z_list.append(all_z.detach().cpu().numpy())
                scores_list.append(scores.detach().cpu().numpy())
                labels_list.append(correct_labels.detach().cpu().numpy())
    
                RMSE_valence = MSE(scores[:,0], correct_labels[:,0])**0.5
                RMSE_arousal = MSE(scores[:,1], correct_labels[:,1])**0.5
    
                rmse_v += RMSE_valence.item(); rmse_a += RMSE_arousal.item()
                cnt = cnt + 1

        if n_iter % opt.print_check == 0:
            print('\n[INFO] Images and features for EVAL save.')
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

        self.encoder.train()
        self.regressor.train()

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

                data, emotions, path = data_i['image'], data_i['va'], data_i['path']
        
                valence = np.expand_dims(np.asarray(emotions[0]), axis=1)  # [64, 1]
                arousal = np.expand_dims(np.asarray(emotions[1]), axis=1)
                emo_np = np.concatenate([valence, arousal], axis=1)
                emotions = torch.from_numpy(emo_np).float()
        
                inputs, correct_labels = Variable(data.cuda(),requires_grad=True), Variable(emotions.cuda(),requires_grad=True)

                self.e_lr_scheduler.step()
                self.r_lr_scheduler.step()
        
                self.e_opt.zero_grad()
                self.r_opt.zero_grad()

                if opt.model == 'mlpmixer':
                    _, z = self.encoder(inputs)
                    scores, all_z = self.regressor(z)
                else:
                    z = self.encoder(inputs)
                    scores, all_z = self.regressor(z)

                # calculate loss by parts
                pcc_loss, ccc_loss, ccc_v, ccc_a = pcc_ccc_loss(correct_labels, all_scores)

                inv_loss = MSE(scores[:,0], correct_labels[:,0]) + MSE(scores[:,1], correct_labels[:,1])

                total_loss = inv_loss + 0.5 * (pcc_loss + ccc_loss)
                total_loss.backward()

                self.e_opt.step()
                self.r_opt.step()

                if opt.online_tracker:
                    wandb.log({
                        "total_loss": total_loss.item(),
                        "Inv_loss": inv_loss.item(),
                        "Enc_lr": aa, "Reg_lr": bb,
                        "epoch": epoch, "ccc_v": ccc_v.item(), "ccc_a": ccc_a.item(),
                    })
    
                if n_iter % opt.print_check == 0 and n_iter > 0:
                    torch.save(self.encoder.state_dict(),   opt.save_path+'encoder_{}_{}.t7'.format(n_iter, epoch))
                    torch.save(self.regressor.state_dict(), opt.save_path+'regressor_{}_{}.t7'.format(n_iter, epoch))
                    self.validate(current_info, n_iter)

                n_iter = n_iter + 1
