import os, argparse
from time  import gmtime, strftime
from .baseline import Baseline
from .caf  import CAF
from .avce import AVCE
from .elim import ELIM

from fabulous.color import fg256


class FER(object):

    def __init__(self, opt):
        self.opt = opt
        ifif opt.method == 'elim':
            self.fer = ELIM(opt)
        elif opt.method == 'avce':
            self.fer = AVCE(opt)
        elif opt.method == 'caf':
            self.fer = CAF(opt)
        elif opt.method == 'baseline':
            self.fer == Baseline(opt)
        self.fer.set_loader()
        self.fer.set_model()

    @staticmethod
    def parser():
        app = argparse.ArgumentParser(description='FER-ZOO')

        # DEFAULT SETTINGS
        app.add_argument("--project_title", type=str, default='facial_expression_recognition_zoo', help='Title for wandb.')
        app.add_argument("--method", type=str, default='eif', choices=['elim', 'avce', 'caf', 'baseline'])
        app.add_argument("--freq", type=int, default=1, help='Saving frequency.')
        app.add_argument("--online_tracker", type=int, default=1, help='On(1) or Off(0).')
        app.add_argument("--dataset", type=str, default='aff_wild', help='aff_wild / aff_wild2 / afew_va / affectNet.')
        app.add_argument("--model", type=str, default='alexnet', help='alexnet / resnet18 / mlpmixer.')
        app.add_argument("--vit_path", type=str, default='../mlpmixer_checkpoint/INet_1K/Mixer-B_16.npz')

        # TRAINING
        app.add_argument("--restore_iter", type=int, default=0)
        app.add_argument("--local_rank", type=int, default=0)
        app.add_argument("--amp", action='store_true')
        app.add_argument("--num_epoch", type=int, default=10)
        app.add_argument("--print_check", type=int, default=100)

        app.add_argument("--latent_dim", type=int, default=512)
        app.add_argument("--output_id_dim", type=int, default=501)
        app.add_argument("--id_balance", type=int, default=0)
        app.add_argument("--output_va_dim", type=int, default=2)
        app.add_argument("--epsilon", type=float, default=1e-6)

        # only for ELIM
        app.add_argument("--no_domain", type=int, default=5)
        app.add_argument("--topk", type=int, default=40, help='Minimum length of each ID sample.')
        app.add_argument("--ermfc_input_dim", type=int, default=512)
        app.add_argument("--ermfc_output_dim", type=int, default=2)
        app.add_argument("--domain_sampling", type=str, choices=['gumbel-softmax', 'max-filling', 'none'], default='none')
        app.add_argument("--sinkhorn", type=bool, default=True)
        app.add_argument("--matching_method", type=str, choices=['Sinkhorn', 'Cross-attention'], default='Sinkhorn')
        app.add_argument("--relevance_weighting", type=int, default=1)
        app.add_argument("--warmup_coef1", type=float, default=10, help='Initial warmup phase.')
        app.add_argument("--warmup_coef2", type=float, default=200, help='Real warmup phase.')

        app.add_argument("--e_lr", default=1e-4, type=float)
        app.add_argument("--r_lr", default=1e-4, type=float)
        app.add_argument("--c_lr", default=1e-5, type=float)
        app.add_argument("--f_lr", default=1e-5, type=float)

        app.add_argument("--tr_batch_size", type=int, default=64, help='Mini-batch size for model training.')
        app.add_argument("--te_batch_size", type=int, default=64, help='Mini-batch size for model inference.')
        app.add_argument("--data_path", type=str, default='/PATH/')
        app.add_argument("--save_path", type=str, default='/PATH/')
        return app

    def fit(self):
        opt = self.opt

        current_dir = os.getcwd()
        current_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())

        if opt.online_tracker:
            with open(current_dir+'/log/'+current_time+'.txt', 'w') as f:
                f.writelines(["Title: {} framework (Dataset: {}\t Model: {}).\n".format(opt.method, opt.dataset, opt.model)])
        os.makedirs(opt.save_path, exist_ok=True)
        self.fer.train([current_dir, current_time])
