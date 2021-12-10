import sys
import os
from argparse import ArgumentParser

from pathlib import Path
from utils import Settings, read_csv_in_dict
from dataset import COPDGeneChunkAirway18Labels
from job_runner import ConvEmbeddingExtractor
import random, csv
import pickle, glob


def generate_cross_validation_splits(root_csv, n_fold, seed):
    random.seed(seed)
    output_path = str(Path(root_csv).parent) + '/'
    p_meta, fields = read_csv_in_dict(root_csv, "PatientID")
    fold_size = len(p_meta) // n_fold
    key_list = list(sorted(p_meta.keys()))
    idx_list = random.sample(range(len(key_list)), len(key_list))
    splits = [idx_list[i:i + fold_size] for i in range(0, len(idx_list), fold_size)]
    if len(splits) != n_fold:
        last_ = splits.pop()
        splits[-1].extend(last_)
    for cross_id, te_indices in enumerate(splits):
        tr_indices = [x for x in idx_list if x not in te_indices]
        te_metas = [p_meta[key_list[x]] for x in te_indices]
        tr_metas = [p_meta[key_list[x]] for x in tr_indices]
        print(f"cross :{cross_id}, using {len(te_metas)} te examples and {len(tr_metas)} tr examples.", flush=True)
        cross_folder = os.path.join(output_path, "cross_" + str(cross_id))
        if not os.path.exists(cross_folder):
            os.makedirs(cross_folder)

        with open(cross_folder + '/te.csv', 'wt', newline='') as fp:
            dw = csv.DictWriter(fp, delimiter=',', fieldnames=fields)
            dw.writeheader()
            dw.writerows(x for x in te_metas)

        with open(cross_folder + '/tr.csv', 'wt', newline='') as fp:
            dw = csv.DictWriter(fp, delimiter=',', fieldnames=fields)
            dw.writeheader()
            dw.writerows(x for x in tr_metas)


def generate_tree_data(archive_path, all_csv):
    data_set = COPDGeneChunkAirway18Labels(archive_path, COPDGeneChunkAirway18Labels.get_series_uids(all_csv))
    target_path = os.path.join(archive_path, "derived") + '/conv/'
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    existing_files = glob.glob(target_path + '/*.pkl', recursive=False)
    existing_keys = [Path(f).stem for f in existing_files]
    old_n = len(data_set.series_uids)
    data_set.series_uids = [k for k in data_set.series_uids if k not in existing_keys]
    print(f"old {old_n} now {len(data_set.series_uids)}.")
    for idx, d in enumerate(data_set):
        uid = d['meta']['uid']
        with open(target_path + f'/{uid}.pkl', 'wb') as fp:
            pickle.dump(d, fp)

        print(f"serialized {uid} to {target_path}", flush=True)


def generate_conv_embeddings(archive_path, csv_file, setting_file, ckp_path):
    settings = Settings(setting_file)
    settings.TEST_CSV = csv_file
    settings.DB_PATH = archive_path
    output_path = archive_path + f'/derived/conv_embedding/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    runner = ConvEmbeddingExtractor(settings, ckp_path, output_path)
    runner.run_job()


if __name__ == "__main__":
    print("Docker start running training job.")
    parser = ArgumentParser()
    parser.add_argument('all_csv', type=str, nargs='?',
                        default=r"D:\workspace\datasets\COPDGene\v3\copdgene220/meta_scans.csv",
                        help="list of uids of segmentation maps.")
    parser.add_argument('archive_path', type=str, nargs='?',
                        default=r"D:\workspace\datasets\COPDGene\v3\copdgene220/",
                        help="archive root path.")
    parser.add_argument('n_fold', type=int, nargs='?',
                        default=5,
                        help="set up learning rate.")
    parser.add_argument('seed', type=int, nargs='?',
                        default=555,
                        help="seed.")
    parser.add_argument('--smp', type=str, nargs='?',
                        default=r".\exp_settings/st_cnn.py",
                        help="settings")
    parser.add_argument('--ckp_path', type=str, nargs='?',
                        default=r"D:\workspace\models\cnn/88.pth",
                        help="settings")
    args = parser.parse_args()
    # generate_cross_validation_splits(args.all_csv, args.n_fold, args.seed)
    # generate_tree_data(args.archive_path, args.all_csv)
    generate_conv_embeddings(args.archive_path, args.all_csv, args.smp, args.ckp_path)
