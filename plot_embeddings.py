import sys
import os
from argparse import ArgumentParser

import matplotlib
matplotlib.use('Agg')
from dataset import ConvEmbeddingDataset
from utils import Settings
from utils import get_callable_by_name

def run_training_job(args):
    if args.smp is None:
        setting_module_path = os.path.join(os.path.dirname(__file__), 'st_pgat_spgnn_3.py')
    else:
        setting_module_path = args.smp
    settings = Settings(setting_module_path)
    settings.RELOAD_CHECKPOINT_PATH = args.ckp_path
    settings.LOGGING = settings.PROCESSOR_LOGGING
    runner_cls = get_callable_by_name(settings.PLOT_EMBEDDING_CLS)
    ct = runner_cls(ConvEmbeddingDataset, settings_module=settings)
    ct.run()


if __name__ == "__main__":
    print("Docker start running training job.")
    parser = ArgumentParser()
    parser.add_argument('--smp', type=str, nargs='?',
                        default=None,
                        help="set up scan path.")
    parser.add_argument('--ckp_path', type=str, default=None,
                        help='set checkpoint path.')
    args = parser.parse_args()
    run_training_job(args)
