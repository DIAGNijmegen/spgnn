import sys
import os
from argparse import ArgumentParser

import matplotlib
matplotlib.use('Agg')

from utils import Settings, get_callable_by_name


def run_testing_job(args):
    if args.smp is None:
        setting_module_path = os.path.dirname(__file__) + '/exp_settings/st_gat_1.py'
    else:
        setting_module_path = args.smp
    settings = Settings(setting_module_path)
    settings.RELOAD_CHECKPOINT_PATH = args.ckp_path
    settings.LOGGING = settings.PROCESSOR_LOGGING
    runner_cls = get_callable_by_name(settings.TEST_RUNNER_CLS)
    ct = runner_cls(output_path=args.output_path,
                    settings_module=settings)
    ct.run()


if __name__ == "__main__":
    print("Docker start running testing job.")
    parser = ArgumentParser()
    parser.add_argument('--smp', type=str, nargs='?',
                        default=None,
                        help="set up scan path.")
    parser.add_argument('--output_path', type=str, nargs='?',
                        default=None,
                        help="set up output path.")
    parser.add_argument('--ckp_path', type=str, nargs='?',
                        default=None,
                        help="set up learning rate.")
    args = parser.parse_args()
    run_testing_job(args)