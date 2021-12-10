import os
from argparse import ArgumentParser

import matplotlib
matplotlib.use('Agg')
from utils import Settings, get_callable_by_name

def run_training_job(args):
    if args.smp is None:
        setting_module_path = os.path.dirname(__file__) + '/exp_settings/st_pgat_spgnn_3.py'
    else:
        setting_module_path = args.smp
    settings = Settings(setting_module_path)
    settings.OPTIMIZER['lr'] = args.lr
    settings.RELOAD_CHECKPOINT_PATH = args.cpk_path
    if args.batch_size > 0:
        settings.TRAIN_BATCH_SIZE = args.batch_size
    settings.RELOAD_CHECKPOINT = True if args.pretrain > 0 else False
    runner_cls = get_callable_by_name(settings.JOB_RUNNER_CLS)
    ct = runner_cls(settings)
    ct.run()


if __name__ == "__main__":
    print("Docker start running training job.")
    parser = ArgumentParser()
    parser.add_argument('lr', type=float, nargs='?',
                        default=5e-4,
                        help="set up learning rate.")
    parser.add_argument('pretrain', type=int, nargs='?',
                        default=0,
                        help="if use pretrained model.")
    parser.add_argument('--smp', type=str, nargs='?',
                        default=None,
                        help="set up scan path.")
    parser.add_argument('--cpk_path', type=str, default=None,
                        help='set checkpoint path.')
    parser.add_argument('--batch_size', type=int, default=0,
                        help='set checkpoint path.')
    args = parser.parse_args()
    run_training_job(args)
