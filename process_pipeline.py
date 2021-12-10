import sys
import os
from argparse import ArgumentParser

import matplotlib
matplotlib.use('Agg')

from utils import Settings, get_callable_by_name
from job_runner import SPGNNE2ETest

def run_testing_job(args):
    setting_module_path = os.path.dirname(__file__) + '/exp_settings/st_pgat_spgnn_3.py'
    settings = Settings(setting_module_path)
    settings.RELOAD_CHECKPOINT_PATH = args.ckp_path
    settings.LOGGING = settings.PROCESSOR_LOGGING
    ct = SPGNNE2ETest(input_path=args.input_path, output_path=args.output_path,
                    settings_module=settings, cpk_path=args.ckp_path)
    ct.run()


if __name__ == "__main__":
    print("Docker start running testing job.")
    parser = ArgumentParser()
    parser.add_argument('--input_path', type=str, nargs='?',
                        default=r"D:\workspace\datasets\COPDGene\v3\copdgene220\derived\seg-airways-chunk-labeled/",
                        help="set up output path.")
    parser.add_argument('--output_path', type=str, nargs='?',
                        default=r"D:\workspace\datasets\COPDGene\v3\copdgene220\derived\test/",
                        help="set up output path.")
    parser.add_argument('--ckp_path', type=str, nargs='?',
                        default=None,
                        help="set up learning rate.")
    args = parser.parse_args()
    run_testing_job(args)