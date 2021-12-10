import torch, time
import matplotlib

matplotlib.use('Agg')

import logging.config
import torch.nn.functional as F
import os, sys
import random
import inspect
from dgl import DGLGraph
import glob
import dgl
from matplotlib import pyplot as plt
from collections import defaultdict
import seaborn as sns
from utils import *
from data_sampler import TensorChunkSetLabelFrequencyTypeSampler
from dataset import ChunkCenterBranch, AirwayTreeGraphDataset, ConvEmbeddingDataset, ChunkAirway18LabelsTest
from pathlib import Path
from functools import reduce
from networkx.drawing.nx_pydot import graphviz_layout
from tensorboardX import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import random
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import sparse as sp
from scipy.spatial.distance import euclidean

np.set_printoptions(suppress=True)

LABEL_NAME_MAPPING = {
    1: 'RB1',
    2: 'RB2',
    3: 'RB3',
    4: 'RB4',
    5: 'RB5',
    6: 'RB6',
    7: 'RB7',
    8: 'RB8',
    9: 'RB9',
    10: 'RB10',
    11: 'LB1+2',
    12: 'LB3',
    13: 'LB4',
    14: 'LB5',
    15: 'LB6',
    16: 'LB7+8',
    17: 'LB9',
    18: 'LB10',
    19: 'TRACHEA',
    20: 'RMB',
    21: 'LMB'
}

COLOR_TABLE = {
    1: "#55ff00",
    2: "#ff0000",
    3: "#000cff",
    4: "#ffff00",
    5: "#147d7d",
    6: "#aaaaff",
    7: "#aa557f",
    8: "#55aaff",
    9: "#55ffff",
    10: "#ef07ff",
    11: "#ff8c39",
    12: "#55ffff",
    13: "#155b64",
    14: "#ff5500",
    15: "#aab600",
    16: "#5e6284",
    17: "#93d4ff",
    18: "#da42ff",
    19: "#5eff84",
    20: "#9300ff",
    21: "#da4200",
}


def load_pretrained_model(cpk_path, reload_objects, state_keys, ignored_keys=[], device='cuda'):
    def reload_state(state, reload_dict, overwrite=False, ignored_keys=[]):
        current_dict = state.state_dict()
        if not overwrite:
            saved_dict = {k: v for k, v in reload_dict.items() if k in current_dict}

            # check in saved_dict, some tensors may not match in size.
            matched_dict = {}
            for k, v in saved_dict.items():
                cv = current_dict[k]
                if k in ignored_keys:
                    print(f"ignore key:{k}")
                    continue
                if isinstance(cv, torch.Tensor) and v.size() != cv.size():
                    print(
                        "in {}, saved tensor size {} does not match current tensor size {}"
                            .format(k, v.size(), cv.size()))
                    continue
                matched_dict[k] = v
        else:
            matched_dict = {k: v for k, v in reload_dict.items()}
        current_dict.update(matched_dict)
        state.load_state_dict(current_dict)

    if device == 'cpu':
        saved_states = torch.load(cpk_path, map_location='cpu')
    else:
        saved_states = torch.load(cpk_path)

    min_len = min(len(reload_objects), len(state_keys))
    for n in range(min_len):
        if state_keys[n] in saved_states.keys():
            if state_keys[n] == "metric":
                reload_state(reload_objects[n], saved_states[state_keys[n]], True, ignored_keys)
            else:
                reload_state(reload_objects[n], saved_states[state_keys[n]], False, ignored_keys)
    return saved_states


class JobRunner:
    class ModelMetricState:

        def __init__(self, **kwargs):
            self._state_dict = copy.deepcopy(kwargs)

        def state_dict(self):
            return self._state_dict

        def load_state_dict(self, new_dict):
            self._state_dict.update(new_dict)

    @staticmethod
    def fix_random_seeds(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def make_single_labeled_mask(self, all_labeled_mask, branch_labels):
        all_labeled_mask = all_labeled_mask.squeeze(1)
        single_label_mask = torch.zeros_like(all_labeled_mask).type(all_labeled_mask.type()).float()
        # all branches at least 1.0
        single_label_mask[all_labeled_mask > 0] = 0.5
        for idx, (almask, branch_label) in enumerate(zip(all_labeled_mask,
                                                         branch_labels)):
            center = tuple([ss // 2 for ss in almask.shape])
            assert (int(almask[center]) == int(branch_label))
            single_label_mask[idx][almask == branch_label] = 0.9
        assert (single_label_mask == 0.9).sum().item() == \
               sum((all_labeled_mask[bli] == bl).sum().item() for bli, bl in enumerate(branch_labels))
        return single_label_mask.unsqueeze(1)

    def _prediction_by_branch_probs(self, ref_np, all_labeled_airway_np, branch_centers, branch_probs):
        prediction_np = np.zeros_like(ref_np, dtype=np.uint8)
        prediction_np[ref_np > 0] = 1
        max_v, max_idx = torch.max(branch_probs[:, 1:], 0)
        for label_id, idx in enumerate(max_idx):
            center = branch_centers[idx.item()]
            prediction_np[all_labeled_airway_np == all_labeled_airway_np[center]] = label_id + 2
        return prediction_np

    def __init__(self, setting_module_file_path, settings_module=None, **kwargs):
        if setting_module_file_path is None:
            file_path = Path(inspect.getfile(self.__class__)).as_posix()
            setting_module_file_path = os.path.join(file_path.rpartition('/')[0], "settings.py")

        if settings_module is not None:
            self.settings = settings_module
        else:
            self.settings = Settings(setting_module_file_path)
        # config loggers
        [os.makedirs(x.rpartition('/')[0]) for x in
         get_value_recursively(self.settings.LOGGING, 'filename') if not os.path.exists(x.rpartition('/')[0])]
        logging.config.dictConfig(self.settings.LOGGING)
        self.logger = logging.getLogger(self.settings.EXP_NAME)

        self.exp_path = os.path.join(self.settings.MODEL_ROOT_PATH, self.settings.EXP_NAME) + '/'
        self.debug_path = os.path.join(self.settings.DEBUG_PATH, self.settings.EXP_NAME) + '/'
        if not os.path.exists(self.exp_path):
            os.makedirs(self.exp_path)
        if not os.path.exists(self.debug_path):
            os.makedirs(self.debug_path)
        self.summary_writer = SummaryWriter(log_dir=os.path.join(self.exp_path, "summary"))

        def runner_excepthook(excType, excValue, traceback):
            self.logger.error("Logging an uncaught exception",
                              exc_info=(excType, excValue, traceback))

        self.model_metrics_save_dict = JobRunner.ModelMetricState()
        sys.excepthook = runner_excepthook
        # self.fix_random_seeds(33 if not hasattr(self.settings, 'RANDOM_SEED') else self.settings.RANDOM_SEED)

        with open(self.exp_path + '/settings.txt', 'wt', newline='') as fp:
            fp.write(str(self.settings))

    def device_data(self, *args):
        pass

    def print_model_parameters(self, iter):
        for name, param in self.model.named_parameters():
            name = name.replace(".", "_")
            if param.requires_grad:
                p = param.clone().cpu().data.numpy()
                self.summary_writer.add_histogram(name, p, global_step=iter)
                self.summary_writer.add_scalar("mean_{}".format(name), np.mean(p), global_step=iter)
                self.summary_writer.add_scalar("std_{}".format(name), np.std(p), global_step=iter)

    def init(self):
        # create model, initializer, optimizer, scheduler for training
        #  according to settings

        cls = get_callable_by_name(self.settings.INITIALIZER.pop('method'))
        self.parameter_initializer = cls(**self.settings.INITIALIZER)
        cls = get_callable_by_name(self.settings.MODEL.pop('method'))
        self.model = cls(**self.settings.MODEL)
        if not hasattr(self.model, 'is_cuda'):
            setattr(self.model, 'is_cuda', torch.cuda.is_available())
        self.is_cuda = self.settings.IS_CUDA & torch.cuda.is_available()
        if self.is_cuda:
            self.model = self.model.cuda()
        # plot MAC and memory
        print("version 12.2.0rc")
        # spatial_size = self.settings.RESAMPLE_SIZE if hasattr(self.settings, 'RESAMPLE_SIZE') else (80, 80, 80)
        # macs, params = get_model_complexity_info(self.model, (self.model.in_ch_list[0], *spatial_size),
        #                                          as_strings=True,
        #                                    print_per_layer_stat=True, verbose=True)

        # self.logger.info(f"macs: {macs}, params: {params}")

        if not isinstance(self.model, torch.nn.DataParallel):
            self.model.init(self.parameter_initializer)
        else:
            self.model.module.init(self.parameter_initializer)
        # create an optimizer wrapper according to settings
        cls = get_callable_by_name(self.settings.OPTIMIZER.pop('method'))
        if 'groups' in self.settings.OPTIMIZER.keys():
            self.settings.optimizer_groups_settings = self.settings.OPTIMIZER.pop('groups')
            rest_parameters = [{'params': [param for name, param in self.model.named_parameters()
                                           if not any(
                    key in name for key in self.settings.optimizer_groups_settings.keys())]}]
            self.optimizer = cls(
                [{'params': list(getattr(self.model, key).parameters()), **self.settings.optimizer_groups_settings[key]}
                 for key in self.settings.optimizer_groups_settings.keys()] + rest_parameters,
                **self.settings.OPTIMIZER)
        else:
            self.optimizer = cls(self.model.parameters(), **self.settings.OPTIMIZER)

        # create a loss wrapper according to settings
        cls = get_callable_by_name(self.settings.LOSS_FUNC.pop('method'))
        self.loss_func = cls(**self.settings.LOSS_FUNC)

        # create a scheduler wrapper according to settings
        cls = get_callable_by_name(self.settings.SCHEDULER.pop('method'))
        self.scheduler = cls(self.optimizer, **self.settings.SCHEDULER)

        if hasattr(self.settings, 'USE_GRAD_SCALER'):
            self.scaler = torch.cuda.amp.GradScaler()
        if hasattr(self.settings, "PRECISION"):
            precision_dict = self.settings.PRECISION
            from apex import amp
            if len(precision_dict.keys()) == 0:
                self.amp_module = amp.init(True)
            else:
                self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
                                                            opt_level=precision_dict["OPT"],
                                                            keep_batchnorm_fp32=precision_dict
                                                            .get("KEEPBN32", None),
                                                            loss_scale=precision_dict.get("LOSS_SCALE", None)
                                                            )
                self.amp_module = amp
            self.logger.info("amp init finished, with config = {}.".format(precision_dict))

        else:
            self.logger.info("amp is None, Full 32 mode.")
            self.amp_module = None
        self.logger.debug("init finished, with full config = {}.".format(self.settings))
        self.current_iteration = 0
        self.epoch_n = 0
        self.saved_model_states = {}

    def generate_batches(self, *args):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def run_job(self):
        try:
            self.run()
        except:
            self.logger.exception("training encounter exception.")

    def reload_model_from_cache(self):
        # check if we need to load pre-trained models. user can either specify a checkpoint by
        # giving an absolute path, or let the engine searching for the latest checkpoint in
        # model output path.
        if self.settings.RELOAD_CHECKPOINT:
            if self.settings.RELOAD_CHECKPOINT_PATH is not None:
                cpk_name = self.settings.RELOAD_CHECKPOINT_PATH
            else:
                # we find the checkpoints from the model output path, we reload whatever the newest.
                list_of_files = glob.glob(self.exp_path + '/*.pth')
                if len(list_of_files) == 0:
                    self.logger.error("{} has no checkpoint files with pth extensions."
                                      .format(self.exp_path))
                    return
                cpk_name = max(list_of_files, key=os.path.getctime)
            self.logger.info("reloading model from {}.".format(cpk_name))
            reload_dicts = self.settings.RELOAD_DICT_LIST
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if not hasattr(self.settings, 'RELOAD_CHECKPOINT_IGNORE'):
                ignore_keys = []
            else:
                ignore_keys = self.settings.RELOAD_CHECKPOINT_IGNORE
            self.saved_model_states = load_pretrained_model(cpk_name,
                                                            [self.model, self.model_metrics_save_dict,
                                                             self.optimizer, self.scheduler,
                                                             ],
                                                            reload_dicts, ignore_keys, device)
            if hasattr(self.settings, 'PRECISION'):
                from apex import amp
                amp.load_state_dict(self.saved_model_states['amp'])
            self.current_iteration = self.saved_model_states['iteration'] + 1 \
                if 'iteration' in self.saved_model_states.keys() else 0
            self.epoch_n = self.saved_model_states['epoch_n'] \
                if 'epoch_n' in self.saved_model_states.keys() else 0

    def update_model_state(self, **kwargs):
        self.saved_model_states['iteration'] = self.current_iteration
        self.saved_model_states['epoch_n'] = self.epoch_n
        self.saved_model_states['model_dict'] = self.model.state_dict()
        self.saved_model_states['optimizer_dict'] = self.optimizer.state_dict()
        self.saved_model_states['scheduler_dict'] = self.scheduler.state_dict()
        self.saved_model_states['metric'] = self.model_metrics_save_dict.state_dict()
        if hasattr(self.settings, 'PRECISION'):
            from apex import amp
            self.saved_model_states['amp']: amp.state_dict()
        self.saved_model_states.update(kwargs)

    def save_model(self, **kwargs):
        self.update_model_state(**kwargs)
        cpk_name = os.path.join(self.exp_path, "{}.pth".format(self.current_iteration))
        time.sleep(10)
        torch.save(self.saved_model_states, cpk_name)
        self.logger.info("saved model into {}.".format(cpk_name))

    def archive_results(self, results):
        raise NotImplementedError


def visualize_airway_graph(path, name, graph, labels_mappings, node_color='#1f78b4'):
    plt.figure(figsize=(25, 15))
    pos = graphviz_layout(graph, prog='dot', root=0)
    # pos_dict = {n:graph.nodes[n]['point'][:-1] for n in graph.nodes()}
    # l = nx.spring_layout(graph, dim=2, pos=pos_dict, scale=2.0)
    nx.draw(graph, pos, labels=labels_mappings, with_labels=True, arrows=False,
            node_size=1000, font_weight='bold', font_size=10, node_color=node_color)
    plt.savefig('{}/{}.png'.format(path, name))
    plt.clf()
    plt.close()


class BaselineTrain(JobRunner):

    def __init__(self, settings_module=None):
        super(BaselineTrain, self).__init__(None, settings_module)
        self.init()
        self.reload_model_from_cache()
        self.series_records = {}
        self.training_phase = 0
        self.tr_uids = AirwayTreeGraphDataset.get_series_uids(self.settings.TRAIN_CSV)
        self.val_uids = AirwayTreeGraphDataset.get_series_uids(self.settings.VALID_CSV)

    def reset_data(self):
        self.logger.info("************Here we are at reset data schedule!**************")
        sample_tr_uids = random.sample(self.tr_uids[:], self.settings.TRAIN_SAMPLE_SIZE)
        tr_dataset = AirwayTreeGraphDataset(self.settings.DB_PATH,
                                            sample_tr_uids)
        tr_loader = DataLoader(tr_dataset, shuffle=True,
                               batch_size=1, collate_fn=defaut_collate_func,
                               num_workers=self.settings.NUM_WORKERS)
        val_loader = DataLoader(AirwayTreeGraphDataset(self.settings.DB_PATH, self.val_uids),
                                batch_size=1, collate_fn=defaut_collate_func,
                                num_workers=self.settings.NUM_WORKERS)

        self.logger.info("************Finished reset data schedule!**************")
        return tr_loader, val_loader

    def evaluate_scan(self, batch_data):
        chunk_size = self.settings.TEST_STITCHES_PATCH_SIZE
        chunk_batch_size = self.settings.TEST_BATCH_SIZE
        self.model.eval()
        with torch.no_grad():
            now = time.time()
            ref = batch_data['#reference']
            all_labeled_airway = batch_data['#all_labeled_reference']
            branch_info = batch_data['branch_info']
            scan_meta_batch = batch_data["meta"]
            series_uid = scan_meta_batch['uid'][0]
            ref_np = squeeze_dims(ref, 3).cpu().numpy()
            all_labeled_airway_np = squeeze_dims(all_labeled_airway, 3).cpu().numpy()
            chunked_loader = DataLoader(
                ChunkCenterBranch({
                    "chunked_list": [all_labeled_airway, ref],
                    "meta": {
                        "uid": [series_uid]
                    }
                }, chunk_size, branch_info, all_label_idx=0, ref_idx=-1),
                batch_size=chunk_batch_size,
                num_workers=self.settings.NUM_WORKERS,
                collate_fn=defaut_collate_func, drop_last=False
            )
            num_center_locations = len(chunked_loader.dataset)
            self.logger.info("start running cubes steps. {}, {}.".format(len(chunked_loader),
                                                                         num_center_locations))
            chunk_outs_list = []
            chunk_center_list = []
            for cube_idx, chunked_ in enumerate(chunked_loader):
                r_batch_all, *ignored = chunked_['chunked_list']
                r_batch_all = r_batch_all.unsqueeze(1)
                chunk_meta = chunked_['meta']
                chunk_infos = search_dict_key_recursively(chunk_meta, 'meta', 'info')[0]
                chunk_labels = [chunk_info[-1] for chunk_info in chunk_infos]
                chunk_centers = search_dict_key_recursively(chunk_meta, 'meta', 'target_center')[0]
                r_batch_single_label = self.make_single_labeled_mask(r_batch_all, chunk_labels).float()
                if self.is_cuda and torch.cuda.is_available():
                    r_batch_single_label = r_batch_single_label.cuda()
                self.model.zero_grad()
                chunk_outs = self.model(r_batch_single_label)
                for cc, co in zip(chunk_centers, chunk_outs):
                    chunk_outs_list.append(co.view(-1))
                    chunk_center_list.append(cc)
            chunk_outs_list = torch.stack(chunk_outs_list, dim=0)
            chunk_infers = F.softmax(chunk_outs_list, dim=1)
            prediction_np = self._prediction_by_branch_probs(ref_np, all_labeled_airway_np,
                                                             chunk_center_list, chunk_infers)

            original_spacing = scan_meta_batch['original_spacing']
            original_size = scan_meta_batch['original_size']
            spacing = scan_meta_batch['spacing']
            original_spacing = np.asarray(original_spacing).flatten().tolist()
            original_size = np.asarray(original_size).flatten().tolist()
            spacing = np.asarray(spacing).flatten().tolist()
            prediction_np, _ = resample(prediction_np, spacing, factor=2, required_spacing=original_spacing,
                                        new_size=original_size, interpolator='nearest')
            ref_np, _ = resample(ref_np, spacing, factor=2, required_spacing=original_spacing,
                                 new_size=original_size, interpolator='nearest')
            gtd_labels, pred_labels = calculate_object_labels(prediction_np, ref_np,
                                                              list(range(2, self.settings.EVAL_NR_CLASS + 2)))
            scan_acc = accuracy_score(gtd_labels, pred_labels)
            end = time.time()
            elapse = end - now
            self.logger.info("VAL Finished scans {}, in {} seconds, {} acc."
                             .format(series_uid, elapse, scan_acc))
            return scan_acc, elapse

    def train(self, loader):
        chunk_size = self.settings.TRAIN_STITCHES_PATCH_SIZE
        chunk_batch_size = self.settings.TRAIN_BATCH_SIZE
        chunk_sample_rate = self.settings.TRAIN_CHUNK_SAMPLE_RATE
        for scan_idx, batch_data in enumerate(loader):
            airway = batch_data['#reference']
            all_labeled_airway = batch_data['#all_labeled_reference']
            branch_info = batch_data['branch_info']

            scan_meta_batch = batch_data["meta"]
            uid = scan_meta_batch['uid'][0]
            chunk_dset = ChunkCenterBranch({
                "chunked_list": [all_labeled_airway, airway],
                "meta": {
                    "uid": [uid]
                }
            }, chunk_size, branch_info, all_label_idx=0, ref_idx=-1)
            chunked_loader = DataLoader(
                chunk_dset,
                sampler=TensorChunkSetLabelFrequencyTypeSampler(chunk_dset, chunk_sample_rate, None),
                batch_size=chunk_batch_size,
                num_workers=self.settings.NUM_WORKERS, collate_fn=defaut_collate_func,
                drop_last=False
            )
            for chunked_ in chunked_loader:
                chunk_meta = chunked_['meta']
                r_batch_all, r_batch = chunked_['chunked_list']
                r_batch_all = r_batch_all.unsqueeze(1)
                r_batch = r_batch.unsqueeze(1)
                self.step((r_batch_all, r_batch, chunk_meta))

    def step(self, train_batch):
        self.model.train()
        cw = [self.settings.CLASS_WEIGHTS[k] for k in sorted(self.settings.CLASS_WEIGHTS.keys())][1:]
        with torch.set_grad_enabled(True):
            r_batch_all, r_batch, batch_meta = train_batch
            chunk_types = search_dict_key_recursively(batch_meta, 'meta', 'type')[0]
            chunk_infos = search_dict_key_recursively(batch_meta, 'meta', 'info')[0]
            branch_labels = [chunk_info[-1] for chunk_info in chunk_infos]
            r_batch_single_label = self.make_single_labeled_mask(r_batch_all, branch_labels).float()
            self.optimizer.zero_grad()
            chunk_g_ref = torch.cat([torch.LongTensor([cl]) for cl in chunk_types], 0) - 1.0
            if torch.cuda.is_available() and self.model.is_cuda:
                chunk_g_ref = chunk_g_ref.cuda()
                r_batch_single_label = r_batch_single_label.cuda()
            batch_output = self.model(r_batch_single_label)
            batch_output = batch_output.view(batch_output.shape[:2])
            total_loss = F.cross_entropy(batch_output, chunk_g_ref.long(),
                                         weight=torch.Tensor([cw]).type(batch_output.type()))
            total_loss.backward()
            self.optimizer.step()

            if self.current_iteration % self.settings.LOG_STEPS == 0:
                step_dict = {
                    "train_loss": total_loss.item(),
                }
                self.summary_writer.add_scalars("train_loss", step_dict, global_step=self.current_iteration)
                self.logger.info(f"Iter {self.epoch_n}/{self.current_iteration} step_dict {step_dict}.")
                self.model_metrics_save_dict.load_state_dict(step_dict)
            self.current_iteration += 1

    def update_epoch(self):
        self.scheduler.step()
        self.epoch_n += 1

    def validate(self, loader):
        self.logger.info("\r\n************At {}, we save states by validating {} scans.**************\r\n"
                         .format(self.current_iteration, len(loader.dataset)))

        self.model.eval()

        average_time = 0
        acc_mean = 0
        for scan_batch_idx, batch_data in enumerate(loader):
            m_acc, elapse = self.evaluate_scan(batch_data)
            acc_mean += m_acc
            average_time += elapse
        acc_mean /= len(loader)
        self.logger.info("\r\nVAL ACC:{}\r\n".format(acc_mean))
        self.model_metrics_save_dict.load_state_dict({"val_acc": acc_mean})
        return acc_mean

    def run(self):

        self.logger.info("start running iterations from {} to {}. "
                         "We have {} total training scan and {} valid scans."
                         .format(self.epoch_n, self.settings.NUM_EPOCHS, len(self.tr_uids),
                                 len(self.val_uids)))
        while self.epoch_n < self.settings.NUM_EPOCHS:
            tr_loader, val_loader = self.reset_data()
            self.train(tr_loader)
            if (self.epoch_n % self.settings.SAVE_EPOCHS == 0 and self.epoch_n > 0) \
                    or self.epoch_n == self.settings.NUM_EPOCHS - 1:
                self.validate(val_loader)
                self.update_model_state()
                self.save_model()
            self.update_epoch()
            self.logger.info(f"Epoch {self.epoch_n} Finished.")

        self.logger.info("Training Finished at {}".format(self.current_iteration))


class BaselineTest(JobRunner):

    def __init__(self, settings_module=None, cpk_path=None, output_path=None):
        super(BaselineTest, self).__init__(None, settings_module)
        test_uids = AirwayTreeGraphDataset.get_series_uids(self.settings.TEST_CSV)
        self.dataset = AirwayTreeGraphDataset(self.settings.DB_PATH, test_uids)
        self.settings.RELOAD_CHECKPOINT = True
        if cpk_path is not None:
            self.settings.RELOAD_CHECKPOINT_PATH = cpk_path
        self.init()
        self.reload_model_from_cache()
        self.output_path = output_path

    def run(self):
        self.logger.info("Start testing {} scans."
                         .format(len(self.dataset)))
        self.model.eval()
        reversed_relabel_mapping = {v: k for k, v in self.settings.RELABEL_MAPPING.items()}
        if self.output_path is None:
            self.output_path = os.path.join(self.exp_path, "{:d}_{:d}"
                                            .format(self.epoch_n, self.current_iteration))
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            with open(self.output_path + '/settings.txt', 'wt', newline='') as fp:
                fp.write(str(self.settings))

        self.logger.info("Start testing {} scans after exclusion."
                         .format(len(self.dataset)))
        average_time = []
        average_acc = AverageMeter()
        chunk_size = self.settings.TEST_STITCHES_PATCH_SIZE
        chunk_batch_size = self.settings.TEST_BATCH_SIZE
        try:
            with torch.no_grad():
                for idx, batch_data in enumerate(self.dataset):
                    now = time.time()
                    ref = batch_data['#reference']
                    all_labeled_airway = batch_data['#all_labeled_reference']
                    branch_info = batch_data['branch_info']
                    scan_meta_batch = batch_data["meta"]
                    series_uid = scan_meta_batch['uid']
                    ref_np = squeeze_dims(ref, 3).cpu().numpy()
                    all_labeled_airway_np = squeeze_dims(all_labeled_airway, 3).cpu().numpy()

                    chunked_loader = DataLoader(
                        ChunkCenterBranch({
                            "chunked_list": [all_labeled_airway.unsqueeze(0), ref.unsqueeze(0)],
                            "meta": {
                                "uid": [series_uid]
                            }
                        }, chunk_size, [[x] for x in branch_info], all_label_idx=0, ref_idx=-1),
                        batch_size=chunk_batch_size,
                        num_workers=self.settings.NUM_WORKERS,
                        collate_fn=defaut_collate_func, drop_last=False
                    )
                    num_center_locations = len(chunked_loader.dataset)
                    self.logger.info("start running cubes steps. {}, {}.".format(len(chunked_loader),
                                                                                 num_center_locations))
                    chunk_outs_list = []
                    chunk_center_list = []
                    for cube_idx, chunked_ in enumerate(chunked_loader):
                        r_batch_all, *ignored = chunked_['chunked_list']
                        r_batch_all = r_batch_all.unsqueeze(1)
                        chunk_meta = chunked_['meta']
                        chunk_infos = search_dict_key_recursively(chunk_meta, 'meta', 'info')[0]
                        chunk_ids = [chunk_info[-1] for chunk_info in chunk_infos]
                        chunk_centers = search_dict_key_recursively(chunk_meta, 'meta', 'target_center')[0]
                        r_batch_single_label = self.make_single_labeled_mask(r_batch_all, chunk_ids).float()
                        if self.is_cuda and torch.cuda.is_available():
                            r_batch_single_label = r_batch_single_label.cuda()
                        chunk_outs = self.model(r_batch_single_label)
                        for cc, co in zip(chunk_centers, chunk_outs):
                            chunk_outs_list.append(co.view(-1))
                            chunk_center_list.append(cc)
                    chunk_outs_list = torch.stack(chunk_outs_list, dim=0)
                    chunk_infers = F.softmax(chunk_outs_list, dim=1)
                    prediction_np = self._prediction_by_branch_probs(ref_np, all_labeled_airway_np,
                                                                     chunk_center_list, chunk_infers)
                    end = time.time()
                    prediction_np = relabel(prediction_np, reversed_relabel_mapping)
                    ref_np = relabel(ref_np, reversed_relabel_mapping)
                    original_spacing = scan_meta_batch['original_spacing']
                    original_size = scan_meta_batch['original_size']
                    spacing = scan_meta_batch['spacing']
                    original_spacing = np.asarray(original_spacing).flatten().tolist()
                    original_size = np.asarray(original_size).flatten().tolist()
                    spacing = np.asarray(spacing).flatten().tolist()
                    prediction_np, _ = resample(prediction_np, spacing, factor=2, required_spacing=original_spacing,
                                                new_size=original_size, interpolator='nearest')
                    ref_np, _ = resample(ref_np, spacing, factor=2, required_spacing=original_spacing,
                                         new_size=original_size, interpolator='nearest')

                    gtd_labels, pred_labels = calculate_object_labels(prediction_np, ref_np,
                                                                      list(range(2, self.settings.EVAL_NR_CLASS + 2)))
                    scan_acc = accuracy_score(gtd_labels, pred_labels)

                    self.logger.info("prediction_np contains: {}".format(np.unique(prediction_np)[1:]))
                    series_archive_path = self.archive_results(
                        (prediction_np,
                         ref_np, all_labeled_airway_np,
                         scan_meta_batch))
                    elapse = end - now
                    average_time.append(elapse)
                    average_acc.update(scan_acc, 1)
                    self.logger.info(f"Finished {idx} test scans ACC {scan_acc}, "
                                     f"archived results in {series_archive_path}, in {elapse} seconds.")

        except StopIteration:
            pass
        finally:

            self.logger.info(f"Finished testing, average time = {np.mean(average_time)}, "
                             f"std time: {np.std(average_time)}, "
                             f"avg_acc: {average_acc.avg}")

    def archive_results(self, test_results):

        predictions, ref, all_labeled_map, scan_meta = test_results
        # from branch info to skeleton map
        series_uid = scan_meta['uid']
        original_spacing = scan_meta['original_spacing']
        original_size = scan_meta['original_size']
        spacing = scan_meta['spacing']
        original_spacing = np.asarray(original_spacing).flatten().tolist()
        original_size = np.asarray(original_size).flatten().tolist()
        origin = np.asarray(scan_meta["origin"]).flatten().tolist()
        direction = np.asarray(scan_meta["direction"]).flatten().tolist()
        spacing = np.asarray(spacing).flatten().tolist()
        assert (predictions.shape == tuple(original_size))
        write_array_to_mhd_itk(self.output_path, [predictions], [series_uid], type=np.uint8,
                               origin=origin[::-1],
                               direction=np.asarray(direction).reshape(3, 3)[
                                         ::-1].flatten().tolist(),
                               spacing=original_spacing[::-1])

        # step.3 dump debug screenshots if the flag TEST_RESULTS_DUMP_DEBUG is True
        self.test_dump_num = 0
        series_path = os.path.join(self.output_path, series_uid)
        if self.settings.TEST_RESULTS_DUMP_DEBUG_NUM > 0 and \
                self.test_dump_num < self.settings.TEST_RESULTS_DUMP_DEBUG_NUM:

            if not os.path.exists(series_path):
                os.makedirs(series_path)
            series_debug_path = os.path.join(series_path, "screenshots")
            if not os.path.exists(series_debug_path):
                os.makedirs(series_debug_path)
            self.logger.info("ref_labels:{}, pred_labels:{}".format(np.unique(ref), np.unique(predictions)))
            ref, _ = resample(ref, spacing, factor=2, required_spacing=original_spacing,
                              new_size=original_size, interpolator='nearest')
            tiled_predictions = np.zeros((ref.shape[0] * 2,) + ref.shape[1:], dtype=np.uint8)
            tiled_predictions[:ref.shape[0], ::] = predictions
            tiled_predictions[ref.shape[0]:, ::] = ref
            # top reference, bottom prediction
            write_array_to_mhd_itk(series_debug_path, [tiled_predictions],
                                   ["tiled_prediction"], type=np.uint8,
                                   origin=origin[::-1],
                                   direction=np.asarray(direction).reshape(3, 3)[
                                             ::-1].flatten().tolist(),
                                   spacing=original_spacing[::-1])
            labels = np.unique(ref)[2:]
            self.logger.info("archive results generate screenshots with unique labels {} in prediction."
                             .format(labels))
            self.test_dump_num += 1
        return series_path


class ConvEmbeddingExtractor(BaselineTest):
    def __init__(self, settings_module=None,
                 cpk_path=None, output_path=None):
        super(ConvEmbeddingExtractor, self).__init__(settings_module, cpk_path, output_path)

    def run(self):
        self.logger.info("Start testing {} scans."
                         .format(len(self.dataset)))
        self.model.eval()
        if self.output_path is None:
            self.output_path = os.path.join(self.exp_path, "{:d}_{:d}".format(self.epoch_n, self.current_iteration))

        try:
            chunk_size = self.settings.TEST_STITCHES_PATCH_SIZE
            chunk_batch_size = self.settings.TEST_BATCH_SIZE
            with torch.no_grad():
                for scan_batch_idx, batch_data in enumerate(self.dataset):
                    save_state = {}
                    ref = batch_data['#reference']
                    all_labeled_airway = batch_data['#all_labeled_reference']
                    branch_info = batch_data['branch_info']
                    scan_meta_batch = batch_data["meta"]
                    series_uid = scan_meta_batch['uid']
                    adj = batch_data['adj']
                    chunked_loader = DataLoader(
                        ChunkCenterBranch({
                            "chunked_list": [all_labeled_airway.unsqueeze(0), ref.unsqueeze(0)],
                            "meta": {
                                "uid": [series_uid]
                            }
                        }, chunk_size, [[x] for x in branch_info], all_label_idx=0, ref_idx=-1),
                        batch_size=chunk_batch_size,
                        num_workers=self.settings.NUM_WORKERS,
                        collate_fn=defaut_collate_func, drop_last=False
                    )
                    num_branches = len(branch_info)
                    fvs = torch.zeros((num_branches, self.model.fv_dim))
                    # fvs = torch.zeros((num_branches, 1024))
                    feat_out = torch.zeros((num_branches, self.settings.NR_CLASS))
                    labels = torch.zeros((num_branches,))
                    if torch.cuda.is_available():
                        fvs = fvs.cuda()
                        feat_out = feat_out.cuda()
                        labels = labels.cuda()
                    all_set_check = set()
                    chunk_centers_list = list(range(num_branches))
                    for cube_idx, chunked_ in enumerate(chunked_loader):
                        r_batch_all, *ignored = chunked_['chunked_list']
                        r_batch_all = r_batch_all.unsqueeze(1)
                        chunk_meta = chunked_['meta']
                        chunk_types = search_dict_key_recursively(chunk_meta, 'meta', 'type')[0]
                        chunk_infos = search_dict_key_recursively(chunk_meta, 'meta', 'info')[0]
                        chunk_ids = [chunk_info[-1] for chunk_info in chunk_infos]
                        chunk_centers = search_dict_key_recursively(chunk_meta, 'meta', 'target_center')[0]
                        r_batch_single_label = self.make_single_labeled_mask(r_batch_all, chunk_ids).float()
                        if self.is_cuda and torch.cuda.is_available():
                            r_batch_single_label = r_batch_single_label.cuda()
                        chunk_feats, chunk_outs = self.model.extract_feature(r_batch_single_label)
                        for ct, cc, cf, cl, co in zip(chunk_types, chunk_centers, chunk_feats,
                                                      chunk_ids, chunk_outs):
                            fvs[cl - 1] = cf.view(-1)
                            labels[cl - 1] = ct - 1
                            feat_out[cl - 1] = co.view(-1)
                            chunk_centers_list[cl - 1] = cc
                            all_set_check.add(cl - 1)
                    assert (len(list(all_set_check)) == num_branches)
                    save_state["fvs"] = fvs.cpu().numpy().astype(np.float)
                    save_state["adj"] = adj.astype(np.uint8)
                    save_state["labels"] = labels.cpu().numpy().astype(np.uint8)
                    save_state["fvs_out"] = feat_out.cpu().numpy().astype(np.float)
                    save_state["ref"] = ref.cpu().numpy().astype(np.uint8)
                    save_state["all_airway"] = all_labeled_airway.cpu().numpy().astype(np.int16)
                    save_state["branch_info"] = branch_info
                    save_state["meta"] = scan_meta_batch
                    with open(self.output_path + f'/{series_uid}.pkl', 'wb') as fp:
                        pickle.dump(save_state, fp)

                    self.logger.info(f"{scan_batch_idx}/{len(self.dataset)} "
                                     f"Cov Embedding saved to {self.output_path}/{series_uid}.pkl.")


        except StopIteration:
            self.logger.info("Finished all conv embedding tasks.")


class GCNTest(BaselineTest):

    def __init__(self, settings_module=None, cpk_path=None, output_path=None):
        super(GCNTest, self).__init__(settings_module, cpk_path, output_path)
        test_uids = ConvEmbeddingDataset.get_series_uids(self.settings.TEST_CSV)
        self.dataset = ConvEmbeddingDataset(self.settings.DB_PATH, test_uids, None, True)

    def from_adj_to_graph(self, adj):
        adj_np = adj.cpu().numpy()
        if (np.triu(adj_np) - adj_np).sum() == 0:
            self.logger.info("tree graph, use directional graph instead!")
            G = nx.DiGraph(adj_np)
            G.remove_edges_from(nx.selfloop_edges(G))
        else:
            if self.settings.GRAPH_MODE == "tree_downstream":
                adj_np = np.triu(adj_np)
                G = nx.DiGraph(adj_np)
                G.remove_edges_from(nx.selfloop_edges(G))
            else:
                G = nx.Graph(adj_np)
                G.remove_edges_from(nx.selfloop_edges(G))
        g = DGLGraph(G)
        g.add_edges(g.nodes(), g.nodes())
        return g

    def run(self):
        self.logger.info("Start testing {} scans."
                         .format(len(self.dataset)))
        self.model.eval()
        reversed_relabel_mapping = {v: k for k, v in self.settings.RELABEL_MAPPING.items()}
        if self.output_path is None:
            self.output_path = os.path.join(self.exp_path, "{:d}_{:d}"
                                            .format(self.epoch_n, self.current_iteration))
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            with open(self.output_path + '/settings.txt', 'wt', newline='') as fp:
                fp.write(str(self.settings))

        self.logger.info("Start testing {} scans after exclusion."
                         .format(len(self.dataset)))
        average_time = 0
        avg_acc = AverageMeter()
        try:
            with torch.no_grad():
                for idx, batch_data in enumerate(self.dataset):
                    now = time.time()
                    fvs = torch.from_numpy(batch_data["fvs"]).cuda().float()
                    adj = torch.from_numpy(batch_data["adj"]).cuda().float()

                    ref_np = batch_data["ref"]
                    all_labeled_airway_np = batch_data["all_airway"]
                    scan_meta = batch_data["meta"]
                    chunk_centers_list = [x[0] for x in batch_data['branch_info']]

                    series_uid = scan_meta['uid']
                    g = self.from_adj_to_graph(adj).to(fvs.device)
                    g.ndata['fvs'] = fvs
                    gnn_out, _ = self.model.forward(g)
                    gcn_probs = F.softmax(gnn_out, dim=1)
                    pred_gcn_np = self._prediction_by_branch_probs(ref_np, all_labeled_airway_np,
                                                                   chunk_centers_list, gcn_probs)
                    end = time.time()
                    elapse = end - now
                    ref_np = relabel(ref_np, reversed_relabel_mapping)
                    pred_gcn_np = relabel(pred_gcn_np, reversed_relabel_mapping)
                    original_spacing = scan_meta['original_spacing']
                    original_size = scan_meta['original_size']
                    spacing = scan_meta['spacing']
                    original_spacing = np.asarray(original_spacing).flatten().tolist()
                    original_size = np.asarray(original_size).flatten().tolist()
                    spacing = np.asarray(spacing).flatten().tolist()
                    pred_gcn_np, _ = resample(pred_gcn_np, spacing, factor=2, required_spacing=original_spacing,
                                              new_size=original_size, interpolator='nearest')
                    ref_np, _ = resample(ref_np, spacing, factor=2, required_spacing=original_spacing,
                                         new_size=original_size, interpolator='nearest')

                    gtd_labels, pred_labels = calculate_object_labels(pred_gcn_np, ref_np,
                                                                      list(range(2, self.settings.EVAL_NR_CLASS + 2)))
                    scan_acc_gcn = accuracy_score(gtd_labels, pred_labels)
                    avg_acc.update(scan_acc_gcn, 1)

                    self.logger.info("VAL Finished scans {}, in {} seconds, {} acc_gcn."
                                     .format(series_uid, elapse, scan_acc_gcn))
                    self.logger.info("prediction_np contains: {}".format(np.unique(pred_gcn_np)[1:]))
                    series_archive_path = self.archive_results(
                        (pred_gcn_np,
                         ref_np, all_labeled_airway_np,
                         scan_meta))
                    average_time += elapse
                    self.logger.info("Finished {} test scans, archived results in {}, in {} seconds."
                                     .format(idx, series_archive_path, elapse))

        except StopIteration:
            pass
        finally:
            average_time /= len(self.dataset)
            self.logger.info("Finished testing, average time = {}, avg_acc: {}".format(average_time, avg_acc.avg))


class PlotEmbeddings(JobRunner):
    LABEL_NAME_MAPPING = {
        1: 'RB1',
        2: 'RB2',
        3: 'RB3',
        4: 'RB4',
        5: 'RB5',
        6: 'RB6',
        7: 'RB7',
        8: 'RB8',
        9: 'RB9',
        10: 'RB10',
        11: 'LB1+2',
        12: 'LB3',
        13: 'LB4',
        14: 'LB5',
        15: 'LB6',
        16: 'LB7+8',
        17: 'LB9',
        18: 'LB10',
    }

    def __init__(self, scan_loader_cls, settings_module=None, cpk_path=None, output_path=None):
        super(PlotEmbeddings, self).__init__(None, settings_module)
        self.train_uids = scan_loader_cls.get_series_uids(self.settings.TRAIN_CSV)
        self.test_uids = scan_loader_cls.get_series_uids(self.settings.TEST_CSV)
        trainset = scan_loader_cls(self.settings.DB_PATH, self.train_uids, None, True)
        testaset = scan_loader_cls(self.settings.DB_PATH, self.test_uids, None, True)

        self.dataset = torch.utils.data.ConcatDataset([trainset, testaset])
        self.settings.RELOAD_CHECKPOINT = True
        if cpk_path is not None:
            self.settings.RELOAD_CHECKPOINT_PATH = cpk_path
        self.init()
        self.reload_model_from_cache()
        self.output_path = output_path

    def from_adj_to_graph(self, adj, fvs, fvs_out, labels, idx, series_uid):
        adj_np = adj.cpu().numpy()

        # print graph for verify
        upper_tri_adj = np.triu(adj_np)

        if (upper_tri_adj - adj_np).sum() == 0:
            self.logger.info("tree graph, use directional graph instead!")
            G = nx.DiGraph(adj_np)
            G.remove_edges_from(nx.selfloop_edges(G))
        else:
            if self.settings.GRAPH_MODE == "tree_downstream":
                G = nx.DiGraph(upper_tri_adj)
                G.remove_edges_from(nx.selfloop_edges(G))
            else:
                G = nx.Graph(adj_np)
                G.remove_edges_from(nx.selfloop_edges(G))
        g = DGLGraph(G)
        g = g.to('cuda:0')

        g.ndata['fvs'] = fvs
        g.ndata['fvs_out'] = fvs_out
        g.ndata['y'] = labels
        return g

    def run(self):
        self.logger.info("Start vis embeddings for {} scans."
                         .format(len(self.dataset)))
        self.model.eval()
        # reversed_relabel_mapping = {v: k for k, v in self.settings.RELABEL_MAPPING.items()}
        if self.output_path is None:
            self.output_path = os.path.join(self.exp_path, "{:d}_{:d}"
                                            .format(self.current_iteration,
                                                    self.epoch_n))
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        all_n_embeddings = []
        all_p_embeddings = []
        all_n_feats = []
        all_n_labels = []
        all_sources = []
        all_uids = []
        try:
            with torch.no_grad():
                for idx, save_state in enumerate(self.dataset):
                    fvs = save_state["fvs"].cuda()
                    adj = save_state["adj"].cuda()
                    fvs_out = save_state['fvs_out'].cuda()
                    labels = save_state['labels'].cuda()

                    meta = save_state["meta"]
                    uid = meta['uid']
                    source = "train" if uid in self.train_uids else "test"
                    g = self.from_adj_to_graph(adj, fvs, fvs_out, labels, idx, uid)
                    n_embedding, p_embeddings = self.model.forward_emb(g)
                    all_sources.extend([source] * len(labels))
                    all_uids.extend([uid] * len(labels))
                    all_n_labels.extend([n for n in labels])
                    all_n_feats.extend([n for n in fvs.cpu().numpy()])
                    all_n_embeddings.extend([n for n in n_embedding.cpu().numpy()])
                    all_p_embeddings.extend([n for n in p_embeddings.cpu().numpy()])
                    self.logger.info("Finished scans {}."
                                     .format(uid))

        except StopIteration:
            pass
        finally:
            all_n_labels = np.asarray(all_n_labels).astype(np.uint8)
            r_indices = [idx for idx, n in enumerate(all_n_labels) if n >= 1.0 and n <= 18]
            sources = np.asarray(all_sources)[r_indices]
            all_n_feats_st = StandardScaler().fit_transform(np.asarray(all_n_feats)[r_indices])
            all_n_embeddings_st = StandardScaler().fit_transform(np.asarray(all_n_embeddings)[r_indices])
            all_p_embeddings_st = StandardScaler().fit_transform(np.asarray(all_p_embeddings)[r_indices])
            # all_n_feats_st = np.asarray(all_n_feats)[r_indices]
            # all_n_embeddings_st = np.asarray(all_n_embeddings)[r_indices]
            if all_n_feats_st.shape[1] > 64:
                all_n_feats_st = PCA(n_components=64).fit_transform(all_n_feats_st)
            if all_n_embeddings_st.shape[1] > 64:
                all_n_embeddings_st = PCA(n_components=64).fit_transform(all_n_embeddings_st)
            if all_p_embeddings_st.shape[1] > 64:
                all_p_embeddings_st = PCA(n_components=64).fit_transform(all_p_embeddings_st)
            perplexitys = [35, 15, 50]
            n_iters = [1000, 1000, 1000]
            for perplexity, n_iter in zip(perplexitys, n_iters):
                feats_tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=n_iter).fit_transform(
                    all_n_feats_st)
                embeddings_tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=n_iter).fit_transform(
                    all_n_embeddings_st)
                embeddings_p_tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=n_iter).fit_transform(
                    all_p_embeddings_st)
                current_palette = {LABEL_NAME_MAPPING[k]: COLOR_TABLE[k] for k in COLOR_TABLE}
                sns.set(rc={'figure.figsize': (11.7, 13.27)})
                y = all_n_labels[r_indices]
                for embeddings, title in zip([feats_tsne, embeddings_tsne, embeddings_p_tsne],
                                             ["conv", "struct", "position", "iposition"]):

                    sns.scatterplot(embeddings[:, 0], embeddings[:, 1], hue=[self.LABEL_NAME_MAPPING[yy] for yy in y],
                                    legend='full', style=sources,
                                    palette=current_palette).set_title(title)

                    for label in np.unique(y):
                        indices = np.where(y == label)[0]
                        coor = np.median(embeddings[indices, :], axis=0)
                        plt.text(coor[0], coor[1], f'{self.LABEL_NAME_MAPPING[label]}',
                                 size='medium', color='black', weight='semibold')
                    plt.savefig(self.output_path + f"/{title}_{perplexity}_{n_iter}.png")
                    plt.clf()


class PlotEmbeddingsSPGNN(PlotEmbeddings):

    def __init__(self, scan_loader_cls, settings_module=None, cpk_path=None, output_path=None):
        super(PlotEmbeddingsSPGNN, self).__init__(scan_loader_cls, settings_module,
                                                  cpk_path, output_path)

    def from_adj_to_graph(self, adj, fvs, fvs_outs, labels, batch_id=None, uid=None):
        adj_np = adj.cpu().numpy()
        G = nx.DiGraph(adj_np)

        g = DGLGraph(G)
        g = dgl.remove_self_loop(g)

        g = g.to('cuda:0')

        g.ndata['fvs'] = fvs
        g.ndata['fvs_out'] = fvs_outs
        g.ndata['y'] = labels
        all_pos_encs = self.generate_distant_pos_enc(g, batch_id, uid)
        # self.generate_rw_pos_enc(g)
        # self.compute_eigen_basis(g)

        # if self.trace:
        #     self.distance_sanit_check(g)

        g.add_edges(g.nodes(), g.nodes())
        return g

    def generate_distant_pos_enc(self, g, batch_id, uid):
        n_nodes = g.number_of_nodes()
        # self.collect_common_path(anchors, tree_G, self.settings.POS_ENC_DIM)
        anchors = self.get_anchors_from_cnn_prediction(g, batch_id, uid)
        G = dgl.to_networkx(g.cpu())
        all_pairwise_distances = dict(nx.all_pairs_shortest_path_length(G))
        diameter = nx.algorithms.distance_measures.diameter(G)
        pos_encs = np.empty((n_nodes, len(anchors)), dtype=np.float32)
        all_pos_encs = np.empty((n_nodes, n_nodes), dtype=np.float32)
        for t_node in range(n_nodes):
            dist_dict = all_pairwise_distances[t_node]
            pos_encs[t_node] = np.asarray([dist_dict[a] / float(diameter) for a in anchors])
            all_pos_encs[t_node] = np.asarray([dist_dict[a] / float(diameter) for a in range(n_nodes)])
            # pos_encs[t_node] = np.asarray([1.0 / (dist_dict[a] + 1.0) for a in anchors])
            # all_pos_encs[t_node] = np.asarray([1.0 / (dist_dict[a] + 1.0) for a in range(n_nodes)])

        g.ndata['pos_enc'] = torch.from_numpy(pos_encs).cuda()
        g.ndata['p'] = torch.from_numpy(pos_encs).cuda()
        return torch.from_numpy(all_pos_encs).cuda()

    def get_anchors_from_cnn_prediction(self, g, batch_id, uid):

        n = g.number_of_nodes()
        fvs_out = F.softmax(g.ndata['fvs_out'], dim=1).cpu().numpy()
        y = g.ndata['y'].cpu().numpy()
        anchors = []
        y_hat = np.zeros_like(y)
        mask = np.ones_like(y) * 1.0

        for label in range(1, 22):
            index = np.argmax(fvs_out[:, label] * mask)
            y_hat[index] = label
            mask[index] = 0.0
            anchors.append(index)
        assert len(y_hat.nonzero()[0]) == 21
        A = g.adjacency_matrix().to_dense().numpy()
        acc = np.sum(y_hat == y) / n
        acc_pos = np.sum((y_hat == y)[y_hat.nonzero()]) / len(y.nonzero()[0])
        self.logger.debug(f"batch: {batch_id}, acc: {acc}, acc_pos: {acc_pos}")
        if self.settings.POS_ENC_DIM == 39:
            self.logger.info("use 39 pos_enc")
            Gvis, adding_anchors = self.add_distal_leafs(anchors[:-3], A)
        elif self.settings.POS_ENC_DIM == 21:
            self.logger.info("use 21 pos_enc")
            adding_anchors = []
        else:
            raise NotImplementedError(f"pos enc dim : {self.settings.POS_ENC_DIM}!")
        anchors = anchors + adding_anchors
        return anchors

    def add_distal_leafs(self, anchors, adj_np):
        upper_tri_adj = np.triu(adj_np)
        G = nx.DiGraph(upper_tri_adj)
        G.remove_edges_from(nx.selfloop_edges(G))
        adding_anchors = []
        for anchor in anchors:
            leafs = {n: nx.shortest_path_length(G, anchor, n)
                     for n in nx.descendants(G, anchor) if G.out_degree(n) == 0}
            if len(leafs) == 0:
                adding_anchors.append(anchor)
            else:
                leafs = sorted(leafs.items(), key=lambda x: x[1])
                adding_anchors.append(leafs[-1][0])
        return G, adding_anchors

    def run(self):
        self.logger.info("Start vis embeddings for {} scans."
                         .format(len(self.dataset)))
        self.model.eval()
        # reversed_relabel_mapping = {v: k for k, v in self.settings.RELABEL_MAPPING.items()}
        if self.output_path is None:
            self.output_path = os.path.join(self.exp_path, "{:d}_{:d}"
                                            .format(self.current_iteration,
                                                    self.epoch_n))
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        all_n_embeddings = []
        all_p_embeddings = []
        all_ip_embeddings = []
        all_n_feats = []
        all_n_labels = []
        all_sources = []
        all_uids = []
        try:
            with torch.no_grad():
                for idx, save_state in enumerate(self.dataset):
                    fvs = save_state["fvs"].cuda()
                    adj = save_state["adj"].cuda()
                    fvs_out = save_state['fvs_out'].cuda()
                    labels = save_state['labels'].cuda()

                    meta = save_state["meta"]
                    uid = meta['uid']
                    source = "train" if uid in self.train_uids else "test"
                    g = self.from_adj_to_graph(adj, fvs, fvs_out, labels, idx, uid)
                    n_embedding, p_embeddings = self.model.forward_emb(g)
                    pos_enc = g.ndata['pos_enc']
                    all_sources.extend([source] * len(labels))
                    all_uids.extend([uid] * len(labels))
                    all_n_labels.extend([n for n in labels])
                    all_n_feats.extend([n for n in fvs.cpu().numpy()])
                    all_ip_embeddings.extend([n for n in pos_enc.cpu().numpy()])
                    all_n_embeddings.extend([n for n in n_embedding.cpu().numpy()])
                    all_p_embeddings.extend([n for n in p_embeddings.cpu().numpy()])
                    self.logger.info("Finished scans {}."
                                     .format(uid))

        except StopIteration:
            pass
        finally:

            all_n_labels = np.asarray(all_n_labels).astype(np.uint8)
            r_indices = [idx for idx, n in enumerate(all_n_labels) if n >= 1.0 and n <= 18]
            sources = np.asarray(all_sources)[r_indices]
            all_n_feats_st = StandardScaler().fit_transform(np.asarray(all_n_feats)[r_indices])
            all_n_embeddings_st = StandardScaler().fit_transform(np.asarray(all_n_embeddings)[r_indices])
            all_p_embeddings_st = StandardScaler().fit_transform(np.asarray(all_p_embeddings)[r_indices])
            all_ip_embeddings_st = StandardScaler().fit_transform(np.asarray(all_ip_embeddings)[r_indices])
            # all_n_feats_st = np.asarray(all_n_feats)[r_indices]
            # all_n_embeddings_st = np.asarray(all_n_embeddings)[r_indices]
            if all_n_feats_st.shape[1] > 32:
                all_n_feats_st = PCA(n_components=32).fit_transform(all_n_feats_st)
            if all_n_embeddings_st.shape[1] > 32:
                all_n_embeddings_st = PCA(n_components=32).fit_transform(all_n_embeddings_st)
            if all_p_embeddings_st.shape[1] > 32:
                all_p_embeddings_st = PCA(n_components=32).fit_transform(all_p_embeddings_st)
            if all_p_embeddings_st.shape[1] > 32:
                all_ip_embeddings_st = PCA(n_components=32).fit_transform(all_ip_embeddings_st)
            perplexitys = [35, 15, 50]
            n_iters = [1000, 1000, 1000]
            for perplexity, n_iter in zip(perplexitys, n_iters):
                feats_tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=n_iter).fit_transform(
                    all_n_feats_st)
                embeddings_tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=n_iter).fit_transform(
                    all_n_embeddings_st)
                embeddings_p_tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=n_iter).fit_transform(
                    all_p_embeddings_st)
                embeddings_ip_tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity,
                                          n_iter=n_iter).fit_transform(
                    all_ip_embeddings_st)
                current_palette = {LABEL_NAME_MAPPING[k]: COLOR_TABLE[k] for k in COLOR_TABLE}
                sns.set(rc={'figure.figsize': (11.7, 13.27)})
                y = all_n_labels[r_indices]
                for embeddings, title in zip([feats_tsne, embeddings_tsne, embeddings_p_tsne, embeddings_ip_tsne],
                                             ["conv", "struct", "position", "iposition"]):

                    sns.scatterplot(embeddings[:, 0], embeddings[:, 1], hue=[self.LABEL_NAME_MAPPING[yy] for yy in y],
                                    legend='full', style=sources,
                                    palette=current_palette).set_title(title)

                    for label in np.unique(y):
                        indices = np.where(y == label)[0]
                        coor = np.median(embeddings[indices, :], axis=0)
                        plt.text(coor[0], coor[1], f'{self.LABEL_NAME_MAPPING[label]}',
                                 size='medium', color='black', weight='semibold')
                    plt.savefig(self.output_path + f"/{title}_{perplexity}_{n_iter}.png")
                    plt.clf()


class GCNTrain(JobRunner):

    def __init__(self, settings_module=None):
        super(GCNTrain, self).__init__(None, settings_module)
        self.init()
        self.reload_model_from_cache()
        self.series_records = {}

        self.tr_uids = ConvEmbeddingDataset.get_series_uids(self.settings.TRAIN_CSV)
        self.val_uids = ConvEmbeddingDataset.get_series_uids(self.settings.VALID_CSV)
        self.trace = False
        self.logger.info(self.model)

    def get_class_weights(self, label_tensor, nr_classes):
        ls, lsc = np.unique(label_tensor.cpu().numpy(), return_counts=True)
        class_weights = (1.0 / lsc) / np.sum((1.0 / lsc))

        weight_t = torch.ones((nr_classes,)).float().cuda()
        for l, w in zip(ls, class_weights):
            weight_t[l] = w

        return weight_t

    def evaluate_scan(self, batch_data):
        self.model.eval()
        with torch.no_grad():
            now = time.time()
            adj = torch.from_numpy(batch_data['adj'][0]).cuda().float()
            fvs = torch.from_numpy(batch_data['fvs'][0]).cuda().float()
            fvs_out = torch.from_numpy(batch_data['fvs_out'][0]).cuda().float()
            labels = torch.from_numpy(batch_data['labels'][0]).cuda().long()

            g = self.from_adj_to_graph(adj).to(fvs.device)

            ref_np = batch_data['ref'][0]
            all_labeled_airway_np = batch_data['all_airway'][0]
            meta = batch_data["meta"]
            chunk_centers_list = [x[0] for x in batch_data['branch_info'][0]]
            series_uid = meta['uid'][0]
            fvs_probs = F.softmax(fvs_out, dim=1)
            pred_np_no_gcn = self._prediction_by_branch_probs(ref_np, all_labeled_airway_np,
                                                              chunk_centers_list, fvs_probs)
            g.ndata['fvs'] = fvs
            gnn_out, _ = self.model.forward(g)
            gcn_probs = F.softmax(gnn_out, dim=1)
            pred_gcn_np = self._prediction_by_branch_probs(ref_np, all_labeled_airway_np, chunk_centers_list,
                                                           gcn_probs)

            original_spacing = meta['original_spacing']
            original_size = meta['original_size']
            spacing = meta['spacing']
            original_spacing = np.asarray(original_spacing).flatten().tolist()
            original_size = np.asarray(original_size).flatten().tolist()
            spacing = np.asarray(spacing).flatten().tolist()
            pred_np_no_gcn, _ = resample(pred_np_no_gcn, spacing, factor=2, required_spacing=original_spacing,
                                         new_size=original_size, interpolator='nearest')
            ref_np, _ = resample(ref_np, spacing, factor=2, required_spacing=original_spacing,
                                 new_size=original_size, interpolator='nearest')
            pred_gcn_np, _ = resample(pred_gcn_np, spacing, factor=2, required_spacing=original_spacing,
                                      new_size=original_size, interpolator='nearest')
            gtd_labels, pred_labels = calculate_object_labels(pred_np_no_gcn, ref_np,
                                                              list(range(2, self.settings.EVAL_NR_CLASS + 2)))
            scan_acc_no_gcn = accuracy_score(gtd_labels, pred_labels)
            gtd_labels, pred_labels = calculate_object_labels(pred_gcn_np, ref_np,
                                                              list(range(2, self.settings.EVAL_NR_CLASS + 2)))
            scan_acc_gcn = accuracy_score(gtd_labels, pred_labels)
            end = time.time()
            elapse = end - now
            self.logger.info("VAL Finished scans {}, in {} seconds, {} acc_gcn. {} acc_no_gcn"
                             .format(series_uid, elapse, scan_acc_gcn, scan_acc_no_gcn))
            return scan_acc_gcn, scan_acc_no_gcn, elapse

    def from_adj_to_graph(self, adj, labels_mappings={}, uid=None):
        adj_np = adj.cpu().numpy()

        # print graph for verify
        upper_tri_adj = np.triu(adj_np)
        if uid is not None:
            G4vis = nx.DiGraph(upper_tri_adj)
            G4vis.remove_edges_from(nx.selfloop_edges(G4vis))
            visualize_airway_graph(self.epoch_debug_path, uid, G4vis, labels_mappings)

        if (upper_tri_adj - adj_np).sum() == 0:
            self.logger.info("tree graph, use directional graph instead!")
            G = nx.DiGraph(adj_np)
            G.remove_edges_from(nx.selfloop_edges(G))
        else:
            if self.settings.GRAPH_MODE == "tree_downstream":
                G = nx.DiGraph(upper_tri_adj)
                G.remove_edges_from(nx.selfloop_edges(G))
            else:
                G = nx.Graph(adj_np)
                G.remove_edges_from(nx.selfloop_edges(G))

        g = DGLGraph(G)
        g.add_edges(g.nodes(), g.nodes())
        g.to('cuda:0')
        return g

    def update_epoch(self):
        self.scheduler.step()
        self.epoch_n += 1

    def run(self):

        while self.epoch_n < self.settings.NUM_EPOCHS:
            tr_loader, val_loader = self.reset_data()
            self.logger.info(f"Train {self.epoch_n}-{self.current_iteration}, {len(tr_loader.dataset)} "
                             f"training samples, {len(val_loader.dataset)} valid samples.")
            self.train(tr_loader)

            if (self.epoch_n % self.settings.SAVE_EPOCHS == 0 and self.epoch_n > 0) \
                    or self.epoch_n == self.settings.NUM_EPOCHS - 1:
                self.validate(val_loader)
                self.update_model_state()
                self.save_model()
            self.update_epoch()
            self.logger.info(f"Epoch {self.epoch_n} Finished.")

        self.logger.info("Training Finished at {}-{}".format(self.epoch_n, self.current_iteration))

    def train(self, loader):
        self.model.train()
        self.model.set_gcn_only()

        cw = [self.settings.CLASS_WEIGHTS[k] for k in sorted(self.settings.CLASS_WEIGHTS.keys())][1:]
        weight_t = torch.Tensor([cw]).float().cuda()
        sampling_rate = self.settings.SAMPLING_RATE
        with torch.set_grad_enabled(True):
            for idx, train_batch in enumerate(loader):
                adj_list = [torch.from_numpy(x).cuda().float() for x in train_batch['adj']]
                fvs_list = [torch.from_numpy(x).cuda().float() for x in train_batch['fvs']]
                labels_list = [torch.from_numpy(x).cuda().long() for x in train_batch['labels']]
                label_mappings_list = [{idx: l.item() for idx, l in enumerate(label_list)} for label_list in
                                       labels_list]
                g_list = [self.from_adj_to_graph(adj, label_mappings).to(adj.device) for adj, label_mappings
                          in zip(adj_list, label_mappings_list)]

                for g, fvs, labels in zip(g_list, fvs_list, labels_list):
                    g.ndata['fvs'] = fvs
                    g.ndata['y'] = labels

                n_nodes = [g.number_of_nodes() for g in g_list]
                batch_g = dgl.batch(g_list)

                labels_list_cat = batch_g.ndata['y']
                sampling_t = torch.ones_like(labels_list_cat) * sampling_rate
                sampling_t[labels_list_cat.nonzero(as_tuple=True)] = 1.0
                sampling_t = sampling_t.detach().cpu().numpy()
                random_list = np.random.random_sample(self.settings.GCN_STEPS * sum(n_nodes)) \
                    .reshape(self.settings.GCN_STEPS, sum(n_nodes))

                for n in range(self.settings.GCN_STEPS):
                    # for param_group in self.optimizer.param_groups:
                    #     param_group["lr"] = lr_schedule[n]
                    self.optimizer.zero_grad()
                    mask = [rn < sampling_t[ni] for ni, rn in zip(range(sum(n_nodes)), random_list[n])]
                    assert (all([mask[x.item()] for x in labels_list_cat.nonzero()]))
                    gnn_out, _ = self.model(batch_g)
                    loss = F.cross_entropy(gnn_out[mask], labels_list_cat[mask], weight=weight_t)
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.3)
                    if n % self.settings.LOG_STEPS == 0:
                        self.logger.info(f"Step {self.epoch_n:d}-{self.current_iteration:d}, "
                                         f"{np.sum(mask) / sum(n_nodes):.5f} masked ratio, "
                                         f"LOSS: {loss.item():.5f}, LR:{self.optimizer.param_groups[0]['lr']:.5f}."
                                         )
                        self.summary_writer.add_scalar("gcn_loss", loss.item(), global_step=self.current_iteration)
                    self.optimizer.step()
                    self.current_iteration += 1

    def reset_data(self):
        self.logger.info("************Here we are at reset data schedule!**************")
        sample_tr_uids = random.sample(self.tr_uids[:], self.settings.TRAIN_SAMPLE_SIZE)
        tr_dataset = ConvEmbeddingDataset(self.settings.DB_PATH,
                                          sample_tr_uids)
        tr_scan_loader = DataLoader(tr_dataset, drop_last=False,
                                    batch_size=self.settings.TRAIN_BATCH_SIZE, collate_fn=collate_func_nativa,
                                    num_workers=self.settings.NUM_WORKERS)
        val_scan_loader = DataLoader(ConvEmbeddingDataset(self.settings.DB_PATH, self.val_uids),
                                     drop_last=False,
                                     batch_size=self.settings.VAL_BATCH_SIZE, collate_fn=collate_func_nativa,
                                     num_workers=self.settings.NUM_WORKERS)

        self.logger.info("************Finished reset data schedule!**************")
        return tr_scan_loader, val_scan_loader

    def validate(self, loader):
        self.logger.info("\r\n************At EPoch-{}, we save states by validating {} scans.**************\r\n"
                         .format(self.epoch_n, len(loader.dataset)))

        self.model.eval()
        elapse = AverageMeter()
        accuracy_gcn = AverageMeter()
        accuracy_no_gcn = AverageMeter()
        for idx, data in enumerate(loader):
            acc, acc_n, elap = self.evaluate_scan(data)
            accuracy_gcn.update(acc, 1)
            accuracy_no_gcn.update(acc_n, 1)
            elapse.update(elap, 1)

        self.logger.info("VAL ACC_GCN:{}, ACC_NOGCN:{}, time:{}."
                         .format(accuracy_gcn.avg, accuracy_no_gcn.avg, elapse.avg))
        self.model_metrics_save_dict.load_state_dict({
            'val_acc_gcn': accuracy_gcn.avg,
            'val_acc_no_gcn': accuracy_no_gcn.avg,
        })


class GCNTrainSAGE(GCNTrain):
    def __init__(self, settings_module):
        super(GCNTrainSAGE, self).__init__(settings_module)

    def train(self, loader):
        self.model.train()
        self.model.set_gcn_only()

        cw = [self.settings.CLASS_WEIGHTS[k] for k in sorted(self.settings.CLASS_WEIGHTS.keys())][1:]
        weight_t = torch.Tensor([cw]).float().cuda()
        with torch.set_grad_enabled(True):
            for idx, train_batch in enumerate(loader):
                adj_list = [torch.from_numpy(x).cuda().float() for x in train_batch['adj']]
                fvs_list = [torch.from_numpy(x).cuda().float() for x in train_batch['fvs']]
                labels_list = [torch.from_numpy(x).cuda().long() for x in train_batch['labels']]
                label_mappings_list = [{idx: l.item() for idx, l in enumerate(label_list)} for label_list in
                                       labels_list]
                g_list = [self.from_adj_to_graph(adj, label_mappings).to(adj.device) for adj, label_mappings
                          in zip(adj_list, label_mappings_list)]

                for g, fvs, labels in zip(g_list, fvs_list, labels_list):
                    g.ndata['fvs'] = fvs
                    g.ndata['y'] = labels

                batch_g = dgl.batch(g_list)
                batch_g = batch_g.to(labels_list[0].device)
                # sample a bunch of nodes
                for gcn_step in range(self.settings.GCN_STEPS):
                    nids = random.sample(list(range(batch_g.number_of_nodes())),
                                         int(batch_g.number_of_nodes() * self.model.node_sample_rate))
                    # neighbor sampler
                    sampler = dgl.dataloading.MultiLayerNeighborSampler(
                        self.model.node_ks)
                    dataloader = dgl.dataloading.NodeDataLoader(
                        batch_g,
                        nids,
                        sampler,
                        device=labels_list[0].device,
                        batch_size=self.settings.NODE_BATCH_SIZE,
                        shuffle=True,
                        drop_last=False,
                        num_workers=self.settings.NUM_WORKERS)
                    for n_batch, (input_nodes, seeds, blocks) in enumerate(dataloader):
                        blocks = [block.int().to(labels_list[0].device) for block in blocks]
                        batch_inputs = blocks[0].srcdata['fvs']
                        batch_labels = blocks[-1].dstdata['y']
                        self.optimizer.zero_grad()
                        batch_outputs, _ = self.model.forward_batch(blocks, batch_inputs)
                        loss = F.cross_entropy(batch_outputs, batch_labels, weight=weight_t)
                        loss.backward()
                        self.optimizer.step()

                    if gcn_step % self.settings.LOG_STEPS == 0:
                        self.logger.info(f"Step {self.epoch_n:d}-{self.current_iteration:d}, "
                                         f"LOSS: {loss.item():.5f}, LR:{self.optimizer.param_groups[0]['lr']:.5f}."
                                         )
                        self.summary_writer.add_scalar("gcn_loss", loss.item(), global_step=self.current_iteration)

                    self.current_iteration += 1


class GCNTrainSPGNN(GCNTrain):

    def __init__(self, settings_module):
        super(GCNTrainSPGNN, self).__init__(settings_module)
        self.trace = True
        self.cached_mean_pos_enc = None

    def validate(self, loader):
        self.logger.info("\r\n************At EPoch-{}, we save states by validating {} scans.**************\r\n"
                         .format(self.epoch_n, len(loader.dataset)))

        self.model.eval()
        elapse = AverageMeter()
        accuracy_gcn = AverageMeter()
        all_s_p_embeds = []
        all_p_init_embeds = []
        all_p_embeds = []
        all_labels = []
        for idx, data in enumerate(loader):
            acc, s_p_embeds, p_init_embeds, p_embeds, labels, elap = self.evaluate_scan(data)
            accuracy_gcn.update(acc, 1)
            elapse.update(elap, 1)
            all_s_p_embeds.append(s_p_embeds)
            all_p_init_embeds.append(p_init_embeds)
            all_p_embeds.append(p_embeds)
            all_labels.extend([x for x in labels])
        all_s_p_embeds = np.vstack(all_s_p_embeds)
        all_p_init_embeds = np.vstack(all_p_init_embeds)
        all_p_embeds = np.vstack(all_p_embeds)
        non_zero_indices = np.nonzero(all_labels)[0]
        selected_labels = np.asarray(all_labels)[non_zero_indices]
        # all_s_p_embeds = StandardScaler().fit_transform(all_s_p_embeds[non_zero_indices])
        # all_p_rw_embeds = StandardScaler().fit_transform(all_p_rw_embeds[non_zero_indices])
        # all_p_eigen_embeds = StandardScaler().fit_transform(all_p_eigen_embeds[non_zero_indices])
        # all_p_embeds = StandardScaler().fit_transform(all_p_embeds[non_zero_indices])
        all_s_p_embeds = all_s_p_embeds[non_zero_indices]
        all_p_init_embeds = all_p_init_embeds[non_zero_indices]
        all_p_embeds = all_p_embeds[non_zero_indices]

        if all_s_p_embeds.shape[-1] > 64:
            all_s_p_embeds = PCA(n_components=64).fit_transform(all_s_p_embeds)
        if all_p_init_embeds.shape[-1] > 64:
            all_p_init_embeds = PCA(n_components=64).fit_transform(all_p_init_embeds)
        if all_p_embeds.shape[-1] > 64:
            all_p_embeds = PCA(n_components=64).fit_transform(all_p_embeds)
        current_palette = {LABEL_NAME_MAPPING[k]: COLOR_TABLE[k] for k in COLOR_TABLE}
        epoch_debug_path = self.debug_path + f'/{self.epoch_n}/'
        if not os.path.exists(epoch_debug_path):
            os.makedirs(epoch_debug_path)
        sns.set(rc={'figure.figsize': (11.7, 13.27)})
        for embeds, name in zip([all_s_p_embeds, all_p_init_embeds, all_p_embeds],
                                ["cat", "pos_init", "pos"]):
            embeds_tsne = TSNE(n_components=2, verbose=1, perplexity=15, n_iter=1000).fit_transform(embeds)

            sns.scatterplot(embeds_tsne[:, 0], embeds_tsne[:, 1],
                            hue=[LABEL_NAME_MAPPING[yy] for yy in selected_labels],
                            legend='full',
                            palette=current_palette).set_title('node embedding tSNE')
            for label in np.unique(selected_labels):
                indices = np.where(selected_labels == label)[0]
                coor = np.median(embeds_tsne[indices, :], axis=0)
                plt.text(coor[0], coor[1], f'{LABEL_NAME_MAPPING[label]}',
                         size='medium', color='black', weight='semibold')
            plt.savefig(epoch_debug_path + f"/embed_{name}.png")
            plt.clf()

        self.logger.info("VAL ACC_GCN:{}, time:{}."
                         .format(accuracy_gcn.avg, elapse.avg))
        self.model_metrics_save_dict.load_state_dict({
            'val_acc_gcn': accuracy_gcn.avg,
        })

    def evaluate_scan(self, batch_data):
        self.model.eval()
        with torch.no_grad():
            now = time.time()
            adj = torch.from_numpy(batch_data['adj'][0]).cuda().float()
            fvs = torch.from_numpy(batch_data['fvs'][0]).cuda().float()
            fvs_out = torch.from_numpy(batch_data['fvs_out'][0]).cuda().float()
            labels = torch.from_numpy(batch_data['labels'][0]).cuda().long()

            meta = batch_data["meta"]
            series_uid = meta['uid'][0]
            g = self.from_adj_to_graph(adj, fvs, fvs_out, labels, uid=series_uid, batch_id=0).to(fvs.device)
            g = dgl.batch([g])
            pos_enc = g.ndata['pos_enc']
            # eigvec = g.ndata['eigvec']
            # dist_pos = g.ndata['dist_pos']
            ref_np = batch_data['ref'][0]
            all_labeled_airway_np = batch_data['all_airway'][0]

            chunk_centers_list = [x[0] for x in batch_data['branch_info'][0]]

            gnn_out, n_embed, n_p_embed = self.model.forward(g)
            g.ndata['p'] = n_p_embed
            # g = dgl.remove_self_loop(g)
            # self.distance_sanit_check(g)
            gnn_probs = F.softmax(gnn_out, dim=1)
            pred_gnn_np = self._prediction_by_branch_probs(ref_np, all_labeled_airway_np, chunk_centers_list,
                                                           gnn_probs)
            gtd_labels, pred_labels = calculate_object_labels(pred_gnn_np, ref_np,
                                                              list(range(2, self.settings.EVAL_NR_CLASS + 2)))
            scan_acc_gnn = accuracy_score(gtd_labels, pred_labels)

            end = time.time()
            elapse = end - now
            self.logger.info("VAL Finished scans {}, in {} seconds, {} acc_gnn."
                             .format(series_uid, elapse, scan_acc_gnn))
            return scan_acc_gnn, \
                   n_embed.cpu().numpy(), \
                   pos_enc.cpu().numpy(), \
                   n_p_embed.cpu().numpy(), labels.cpu().numpy(), elapse

    def compute_eigen_basis(self, g):
        pos_enc_dim = self.settings.POS_ENC_DIM
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        L = sp.eye(g.number_of_nodes()) - N * A * N

        # Eigenvectors with numpy
        EigVal, EigVec = np.linalg.eig(L.toarray())
        idx = EigVal.argsort()  # increasing order
        EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
        g.ndata['eigvec'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float().cuda()

        # zero padding to the end if n < pos_enc_dim
        n = g.number_of_nodes()
        if n <= pos_enc_dim:
            g.ndata['eigvec'] = F.pad(g.ndata['eigvec'], (0, pos_enc_dim - n + 1), value=float('0'))

    def distance_sanit_check(self, g):
        # pos_enc = g.ndata['rw_enc'].cpu().numpy()
        # eigvec = g.ndata['eigvec'].cpu().numpy()
        distpos = g.ndata['pos_enc'].cpu().numpy()
        p = g.ndata['p'].cpu().numpy()
        G = g.cpu().to_networkx()
        all_pairwise_distances = dict(nx.all_pairs_shortest_path_length(G))
        diameter = nx.algorithms.distance_measures.diameter(G)
        random_test_nodes = random.sample(range(g.number_of_nodes()), g.number_of_nodes() // 2)
        dists_records = defaultdict(list)
        for t_node in random_test_nodes:
            dist_dict = all_pairwise_distances[t_node]
            for k, v in dist_dict.items():
                dists_records['shortest_path'].append(v / float(diameter))
                # dists_records['shortest_path'].append(1.0 / (1.0 + v) )
                # dists_records['eigenvec'].append(euclidean(eigvec[t_node], eigvec[k]))
                # dists_records['rw_enc'].append(euclidean(pos_enc[t_node], pos_enc[k]))
                dists_records['pos_enc'].append(euclidean(distpos[t_node], distpos[k]))
                dists_records['p'].append(euclidean(p[t_node], p[k]))

        # correlation analysis
        from scipy.stats import spearmanr, pearsonr
        # spear_eigen, _ = spearmanr(dists_records['shortest_path'], dists_records['eigenvec'])
        # spear_rw, _ = spearmanr(dists_records['shortest_path'], dists_records['rw_enc'])
        spear_distpos, _ = spearmanr(dists_records['shortest_path'], dists_records['pos_enc'])
        spear_p, _ = spearmanr(dists_records['shortest_path'], dists_records['p'])
        # pear_eigen, _ = pearsonr(dists_records['shortest_path'], dists_records['eigenvec'])
        # pear_rw, _ = pearsonr(dists_records['shortest_path'], dists_records['rw_enc'])
        pear_distpos, _ = pearsonr(dists_records['shortest_path'], dists_records['pos_enc'])
        pear_p, _ = pearsonr(dists_records['shortest_path'], dists_records['p'])
        self.logger.info(
            # f"spearman coor eigen:{spear_eigen}, rw: {spear_eigen}, "
            f"dist:{spear_distpos}, p:{spear_p}. \r\n"
            # f"pearsonr coor eigen:{pear_eigen}, rw: {pear_rw},"
            f" distpos:{pear_distpos}, p:{pear_p}."
        )

    def generate_rw_pos_enc(self, g):
        n_nodes = g.number_of_nodes()
        # A = g.adjacency_matrix(scipy_fmt="csr")
        A = g.adjacency_matrix().to_dense().numpy()
        # Dinv = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float)  # D^-1
        Dinv = np.eye(n_nodes, dtype=np.float) * (dgl.backend.asnumpy(g.in_degrees()).clip(min=1)) ** -1.0
        RW = np.matmul(A, Dinv)
        M = RW

        # Iterate
        # PE = [torch.from_numpy(M.diagonal()).float()]
        PE = [torch.from_numpy(np.diagonal(M)).float()]
        M_power = M
        for _ in range(self.settings.POS_ENC_DIM - 1):
            M_power = np.matmul(M_power, M)
            # PE.append(torch.from_numpy(M_power.diagonal()).float())
            PE.append(torch.from_numpy(np.diagonal(M_power)).float())
        PE = torch.stack(PE, dim=-1)
        g.ndata['rw_enc'] = PE.cuda()

    def collect_common_path(self, anchors, G, n_common_path):
        leafs = [n for n in G.nodes() if G.out_degree(n) == 0]
        all_paths_leafs = [[p for p in nx.shortest_path(G, source=0, target=n)] for n in leafs]
        al, counts = np.unique([xx for x in all_paths_leafs for xx in x], return_counts=True)
        indices = np.argsort(counts)[-n_common_path:]
        for i in indices:
            anchors.add(al[i])

    def add_distal_leafs(self, anchors, adj_np):
        upper_tri_adj = np.triu(adj_np)
        G = nx.DiGraph(upper_tri_adj)
        G.remove_edges_from(nx.selfloop_edges(G))
        adding_anchors = []
        for anchor in anchors:
            leafs = {n: nx.shortest_path_length(G, anchor, n)
                     for n in nx.descendants(G, anchor) if G.out_degree(n) == 0}
            if len(leafs) == 0:
                adding_anchors.append(anchor)
            else:
                leafs = sorted(leafs.items(), key=lambda x: x[1])
                adding_anchors.append(leafs[-1][0])
        return G, adding_anchors

    def get_anchors_from_cnn_prediction(self, g, batch_id, uid):

        n = g.number_of_nodes()
        fvs_out = F.softmax(g.ndata['fvs_out'], dim=1).cpu().numpy()
        y = g.ndata['y'].cpu().numpy()
        anchors = []
        y_hat = np.zeros_like(y)
        mask = np.ones_like(y) * 1.0

        for label in range(1, 22):
            index = np.argmax(fvs_out[:, label] * mask)
            y_hat[index] = label
            mask[index] = 0.0
            anchors.append(index)
        assert len(y_hat.nonzero()[0]) == 21
        A = g.adjacency_matrix().to_dense().numpy()
        acc = np.sum(y_hat == y) / n
        acc_pos = np.sum((y_hat == y)[y_hat.nonzero()]) / len(y.nonzero()[0])
        self.logger.debug(f"batch: {batch_id}, acc: {acc}, acc_pos: {acc_pos}")
        if self.settings.POS_ENC_DIM == 39:
            Gvis, adding_anchors = self.add_distal_leafs(anchors[:-3], A)
            if self.trace:
                label_mappings = {idx: l for idx, l in enumerate(y_hat)}
                node_colors = ['green' if a in anchors else '#1f78b4' for a in range(n)]
                visualize_airway_graph(self.debug_path, uid + '_anchors', Gvis, label_mappings, node_colors)
        elif self.settings.POS_ENC_DIM == 21:
            adding_anchors = []
        else:
            raise NotImplementedError(f"pos enc dim : {self.settings.POS_ENC_DIM}!")
        anchors = anchors + adding_anchors
        return anchors

    def generate_distant_pos_enc(self, g, batch_id, uid):
        n_nodes = g.number_of_nodes()
        # self.collect_common_path(anchors, tree_G, self.settings.POS_ENC_DIM)
        anchors = self.get_anchors_from_cnn_prediction(g, batch_id, uid)
        G = dgl.to_networkx(g.cpu())
        all_pairwise_distances = dict(nx.all_pairs_shortest_path_length(G))
        diameter = nx.algorithms.distance_measures.diameter(G)
        pos_encs = np.empty((n_nodes, len(anchors)), dtype=np.float32)
        all_pos_encs = np.empty((n_nodes, n_nodes), dtype=np.float32)
        for t_node in range(n_nodes):
            dist_dict = all_pairwise_distances[t_node]
            pos_encs[t_node] = np.asarray([dist_dict[a] / float(diameter) for a in anchors])
            all_pos_encs[t_node] = np.asarray([dist_dict[a] / float(diameter) for a in range(n_nodes)])
            # pos_encs[t_node] = np.asarray([1.0 / (dist_dict[a] + 1.0) for a in anchors])
            # all_pos_encs[t_node] = np.asarray([1.0 / (dist_dict[a] + 1.0) for a in range(n_nodes)])

        g.ndata['pos_enc'] = torch.from_numpy(pos_encs).cuda().float()
        g.ndata['p'] = torch.from_numpy(pos_encs).cuda().float()
        return torch.from_numpy(all_pos_encs).cuda()

    def from_adj_to_graph(self, adj, fvs, fvs_outs, labels, all_pos_encs_cache=None, batch_id=None, uid=None):
        adj_np = adj.cpu().numpy()
        G = nx.DiGraph(adj_np)

        g = DGLGraph(G)
        g = dgl.remove_self_loop(g)

        g = g.to('cuda:0')

        g.ndata['fvs'] = fvs
        g.ndata['fvs_out'] = fvs_outs
        g.ndata['y'] = labels
        all_pos_encs = self.generate_distant_pos_enc(g, batch_id, uid)
        # self.generate_rw_pos_enc(g)
        # self.compute_eigen_basis(g)

        if all_pos_encs_cache is not None:
            all_pos_encs_cache[batch_id] = all_pos_encs
        # if self.trace:
        #     self.distance_sanit_check(g)

        g.add_edges(g.nodes(), g.nodes())
        return g

    def laplacian_pos_loss(self, g):
        total_loss = []
        lamb = self.settings.LAMBDA
        for bg in dgl.unbatch(g):
            p = bg.ndata['p']
            p_zero_sum = p - torch.mean(p, dim=0, keepdim=True).detach()
            p_norm = p_zero_sum / (torch.std(p, dim=0, keepdim=True) + 1e-7).detach()
            # g.ndata['p'] = p_embed
            n = bg.number_of_nodes()

            # Laplacian
            A = bg.adjacency_matrix(scipy_fmt="csr")
            N = sp.diags(dgl.backend.asnumpy(bg.in_degrees()).clip(1) ** -0.5, dtype=float)
            L = sp.eye(n) - N * A * N

            pT = torch.transpose(p_norm, 1, 0)
            loss_b_1 = torch.trace(torch.mm(torch.mm(pT, torch.Tensor(L.todense()).cuda()), p_norm))

            ptp = torch.mm(pT, p_norm) - torch.eye(p_norm.shape[1]).cuda()
            ptp_loss = torch.norm(ptp, p='fro')
            loss = (loss_b_1 + lamb * ptp_loss) / (self.settings.POS_ENC_DIM * n)
            total_loss.append(loss)
        return torch.stack(total_loss).mean()

    def dist_pos_loss(self, g, all_pos_encs_cache):
        total_d_loss = []
        total_c_loss = []
        batch_stats = torch.rand((g.batch_size, self.settings.NR_CLASS - 1,
                                  self.settings.POS_ENC_DIM)).cuda()
        for batch_id, bg in enumerate(dgl.unbatch(g)):
            p = bg.ndata['p']
            y = bg.ndata['y']
            label_mapping = {y[n].item(): n for n in range(y.shape[0]) if y[n].item() != 0}
            existing_keys = list(set(list(range(1, 22))) & set(label_mapping.keys()))
            current_stats = []
            for label in range(1, 22):
                if label in label_mapping.keys():
                    batch_stats[batch_id, label - 1, ::] = p[label_mapping[label]]
                    current_stats.append(p[label_mapping[label]])
            current_stats = torch.stack(current_stats, dim=0)
            if self.cached_mean_pos_enc is not None:
                c_loss = ((current_stats - self.cached_mean_pos_enc[(np.asarray(existing_keys) - 1)]) ** 2).sum()
            else:
                c_loss = torch.FloatTensor([0.0]).cuda()
            n = bg.number_of_nodes()
            all_pos_encs = all_pos_encs_cache[batch_id]
            x = p.unsqueeze(0).repeat(n, 1, 1)
            y = p.unsqueeze(1).repeat(1, n, 1)
            affinity = torch.exp(-1.0 * torch.abs(x - y).sum(dim=2))
            d_loss = F.smooth_l1_loss(affinity, torch.exp(-all_pos_encs))
            total_d_loss.append(d_loss)
            total_c_loss.append(c_loss)
        dist_loss = torch.stack(total_d_loss).mean()
        compact_loss = torch.stack(total_c_loss).mean()
        if self.cached_mean_pos_enc is None:
            self.cached_mean_pos_enc = batch_stats.mean(dim=0).detach()
        else:
            self.cached_mean_pos_enc = 0.15 * self.cached_mean_pos_enc + 0.85 * batch_stats.mean(dim=0).detach()
        return dist_loss, compact_loss

    def train(self, loader):
        self.model.train()
        self.model.set_gcn_only()

        cw = [self.settings.CLASS_WEIGHTS[k] for k in sorted(self.settings.CLASS_WEIGHTS.keys())][1:]
        weight_t = torch.Tensor([cw]).float().cuda()
        sampling_rate = self.settings.SAMPLING_RATE
        with torch.set_grad_enabled(True):
            for idx, train_batch in enumerate(loader):
                adj_list = [torch.from_numpy(x).cuda().float() for x in train_batch['adj']]
                fvs_list = [torch.from_numpy(x).cuda().float() for x in train_batch['fvs']]
                fvs_outs_list = [torch.from_numpy(x).cuda().float() for x in train_batch['fvs_out']]
                labels_list = [torch.from_numpy(x).cuda().long() for x in train_batch['labels']]
                all_pos_encs_cache = {}
                g_list = [
                    self.from_adj_to_graph(adj, fvs, fvs_outs, labels, all_pos_encs_cache, batch_id, uid).to(adj.device)
                    for batch_id, (adj, fvs, labels, fvs_outs, uid)
                    in enumerate(zip(adj_list, fvs_list, labels_list, fvs_outs_list, train_batch['meta']['uid']))]

                batch_g = dgl.batch(g_list)

                n_nodes = [g.number_of_nodes() for g in g_list]
                labels_list_cat = batch_g.ndata['y']
                sampling_t = torch.ones_like(labels_list_cat) * sampling_rate
                sampling_t[labels_list_cat.nonzero(as_tuple=True)] = 1.0
                sampling_t = sampling_t.detach().cpu().numpy()
                random_list = np.random.random_sample(self.settings.GCN_STEPS * sum(n_nodes)) \
                    .reshape(self.settings.GCN_STEPS, sum(n_nodes))

                for n in range(self.settings.GCN_STEPS):
                    # for param_group in self.optimizer.param_groups:
                    #     param_group["lr"] = lr_schedule[n]
                    self.optimizer.zero_grad()
                    mask = [rn < sampling_t[n] for n, rn in zip(range(sum(n_nodes)), random_list[n])]
                    assert (all([mask[x.item()] for x in labels_list_cat.nonzero()]))
                    gnn_out, n_embed, p_embed = self.model(batch_g)
                    batch_g.ndata['p'] = p_embed
                    loss_gnn = F.cross_entropy(gnn_out[mask], labels_list_cat[mask], weight=weight_t)
                    if self.settings.USE_DIST_LOSS:
                        d_loss, c_loss = self.dist_pos_loss(batch_g, all_pos_encs_cache)
                        loss = loss_gnn + d_loss + 0.1 * c_loss
                    else:
                        d_loss, c_loss = torch.FloatTensor([0.0]), torch.FloatTensor([0.0])
                        loss = loss_gnn
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.3)
                    if n % self.settings.LOG_STEPS == 0:
                        self.logger.info(f"Step {self.epoch_n:d}-{self.current_iteration:d}, "
                                         f"{np.sum(mask) / sum(n_nodes):.5f} masked ratio, "
                                         f"LOSS: {loss.item():.5f}, "
                                         f"loss_gnn: {loss_gnn.item():.5f}, "
                                         f"d_loss: {d_loss.item():.5f}, "
                                         f"c_loss: {c_loss.item():.5f}, "
                                         f"LR:{self.optimizer.param_groups[0]['lr']:.5f}."
                                         )
                        self.summary_writer.add_scalar("gcn_loss", loss.item(), global_step=self.current_iteration)
                    self.optimizer.step()
                    self.current_iteration += 1


class GCNTestSPGNN(BaselineTest):

    def __init__(self, settings_module=None, cpk_path=None, output_path=None):
        super(GCNTestSPGNN, self).__init__(settings_module, cpk_path, output_path)
        test_uids = ConvEmbeddingDataset.get_series_uids(self.settings.TEST_CSV)
        self.dataset = ConvEmbeddingDataset(self.settings.DB_PATH, test_uids, None, True)
        self.trace = False

    def from_adj_to_graph(self, adj, fvs, fvs_outs, labels, all_pos_encs_cache=None, batch_id=None, uid=None):
        adj_np = adj.cpu().numpy()
        G = nx.DiGraph(adj_np)

        g = DGLGraph(G)
        g = dgl.remove_self_loop(g)

        g = g.to('cuda:0')

        g.ndata['fvs'] = fvs
        g.ndata['fvs_out'] = fvs_outs
        g.ndata['y'] = labels
        all_pos_encs = self.generate_distant_pos_enc(g, batch_id, uid)
        # self.generate_rw_pos_enc(g)
        # self.compute_eigen_basis(g)

        if all_pos_encs_cache is not None:
            all_pos_encs_cache[batch_id] = all_pos_encs
        # if self.trace:
        #     self.distance_sanit_check(g)

        g.add_edges(g.nodes(), g.nodes())
        return g

    def generate_distant_pos_enc(self, g, batch_id, uid):
        n_nodes = g.number_of_nodes()
        # self.collect_common_path(anchors, tree_G, self.settings.POS_ENC_DIM)
        anchors = self.get_anchors_from_cnn_prediction(g, batch_id, uid)
        G = dgl.to_networkx(g.cpu())
        all_pairwise_distances = dict(nx.all_pairs_shortest_path_length(G))
        diameter = nx.algorithms.distance_measures.diameter(G)
        pos_encs = np.empty((n_nodes, len(anchors)), dtype=np.float32)
        all_pos_encs = np.empty((n_nodes, n_nodes), dtype=np.float32)
        for t_node in range(n_nodes):
            dist_dict = all_pairwise_distances[t_node]
            pos_encs[t_node] = np.asarray([dist_dict[a] / float(diameter) for a in anchors])
            all_pos_encs[t_node] = np.asarray([dist_dict[a] / float(diameter) for a in range(n_nodes)])
            # pos_encs[t_node] = np.asarray([1.0 / (dist_dict[a] + 1.0) for a in anchors])
            # all_pos_encs[t_node] = np.asarray([1.0 / (dist_dict[a] + 1.0) for a in range(n_nodes)])

        g.ndata['pos_enc'] = torch.from_numpy(pos_encs).cuda().float()
        g.ndata['p'] = torch.from_numpy(pos_encs).cuda().float()
        return torch.from_numpy(all_pos_encs).cuda()

    def get_anchors_from_cnn_prediction(self, g, batch_id, uid):

        n = g.number_of_nodes()
        fvs_out = F.softmax(g.ndata['fvs_out'], dim=1).cpu().numpy()
        y = g.ndata['y'].cpu().numpy()
        anchors = []
        y_hat = np.zeros((n,), dtype=np.float)
        mask = np.ones((n,), dtype=np.float)

        for label in range(1, 22):
            index = np.argmax(fvs_out[:, label] * mask)
            y_hat[index] = label
            mask[index] = 0.0
            anchors.append(index)
        assert len(y_hat.nonzero()[0]) == 21
        A = g.adjacency_matrix().to_dense().numpy()
        acc = np.sum(y_hat == y) / n
        acc_pos = np.sum((y_hat == y)[y_hat.nonzero()]) / len(y.nonzero()[0])
        self.logger.debug(f"batch: {batch_id}, acc: {acc}, acc_pos: {acc_pos}")
        Gvis, adding_anchors = self.add_distal_leafs(anchors[:-3], A)
        anchors = anchors + adding_anchors
        if self.trace:
            label_mappings = {idx: l for idx, l in enumerate(y_hat)}
            node_colors = ['green' if a in anchors else '#1f78b4' for a in range(n)]
            visualize_airway_graph(self.debug_path, uid + '_anchors', Gvis, label_mappings, node_colors)
        return anchors

    def add_distal_leafs(self, anchors, adj_np):
        upper_tri_adj = np.triu(adj_np)
        G = nx.DiGraph(upper_tri_adj)
        G.remove_edges_from(nx.selfloop_edges(G))
        adding_anchors = []
        for anchor in anchors:
            leafs = {n: nx.shortest_path_length(G, anchor, n)
                     for n in nx.descendants(G, anchor) if G.out_degree(n) == 0}
            if len(leafs) == 0:
                adding_anchors.append(anchor)
            else:
                leafs = sorted(leafs.items(), key=lambda x: x[1])
                adding_anchors.append(leafs[-1][0])
        return G, adding_anchors

    def run(self):
        self.logger.info("Start testing {} scans."
                         .format(len(self.dataset)))
        self.model.eval()
        reversed_relabel_mapping = {v: k for k, v in self.settings.RELABEL_MAPPING.items()}
        if self.output_path is None:
            self.output_path = os.path.join(self.exp_path, "{:d}_{:d}"
                                            .format(self.epoch_n, self.current_iteration))
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            with open(self.output_path + '/settings.txt', 'wt', newline='') as fp:
                fp.write(str(self.settings))

        self.logger.info("Start testing {} scans after exclusion."
                         .format(len(self.dataset)))
        average_time = 0
        avg_acc = AverageMeter()
        try:
            with torch.no_grad():
                for idx, batch_data in enumerate(self.dataset):
                    now = time.time()
                    adj = torch.from_numpy(batch_data['adj']).cuda().float()
                    fvs = torch.from_numpy(batch_data['fvs']).cuda().float()
                    fvs_out = torch.from_numpy(batch_data['fvs_out']).cuda().float()
                    labels = torch.from_numpy(batch_data['labels']).cuda().float()

                    meta = batch_data["meta"]
                    series_uid = meta['uid']
                    g = self.from_adj_to_graph(adj, fvs, fvs_out, labels, uid=series_uid, batch_id=0).to(fvs.device)
                    g = dgl.batch([g])
                    ref_np = batch_data['ref']
                    all_labeled_airway_np = batch_data['all_airway']

                    chunk_centers_list = [x[0] for x in batch_data['branch_info']]

                    gnn_out, _, _ = self.model.forward(g)
                    gcn_probs = F.softmax(gnn_out, dim=1)
                    pred_gcn_np = self._prediction_by_branch_probs(ref_np, all_labeled_airway_np,
                                                                   chunk_centers_list, gcn_probs)
                    end = time.time()
                    elapse = end - now
                    ref_np = relabel(ref_np, reversed_relabel_mapping)
                    pred_gcn_np = relabel(pred_gcn_np, reversed_relabel_mapping)
                    original_spacing = meta['original_spacing']
                    original_size = meta['original_size']
                    spacing = meta['spacing']
                    original_spacing = np.asarray(original_spacing).flatten().tolist()
                    original_size = np.asarray(original_size).flatten().tolist()
                    spacing = np.asarray(spacing).flatten().tolist()
                    pred_gcn_np, _ = resample(pred_gcn_np, spacing, factor=2, required_spacing=original_spacing,
                                              new_size=original_size, interpolator='nearest')
                    ref_np, _ = resample(ref_np, spacing, factor=2, required_spacing=original_spacing,
                                         new_size=original_size, interpolator='nearest')

                    gtd_labels, pred_labels = calculate_object_labels(pred_gcn_np, ref_np,
                                                                      list(range(2, self.settings.EVAL_NR_CLASS + 2)))
                    scan_acc_gcn = accuracy_score(gtd_labels, pred_labels)
                    avg_acc.update(scan_acc_gcn, 1)

                    self.logger.info("VAL Finished scans {}, in {} seconds, {} acc_gcn."
                                     .format(series_uid, elapse, scan_acc_gcn))
                    self.logger.info("prediction_np contains: {}".format(np.unique(pred_gcn_np)[1:]))
                    series_archive_path = self.archive_results(
                        (pred_gcn_np,
                         ref_np, all_labeled_airway_np,
                         meta))
                    average_time += elapse
                    self.logger.info("Finished {} test scans, archived results in {}, in {} seconds."
                                     .format(idx, series_archive_path, elapse))

        except StopIteration:
            pass
        finally:
            average_time /= len(self.dataset)
            self.logger.info("Finished testing, average time = {}, avg_acc: {}".format(average_time, avg_acc.avg))


class SPGNNE2ETest(JobRunner):

    def __init__(self, input_path, output_path, settings_module=None, cpk_path=None):
        super(SPGNNE2ETest, self).__init__(None, settings_module)
        self.dataset = ChunkAirway18LabelsTest(input_path)
        self.settings.RELOAD_CHECKPOINT = True
        if cpk_path is not None:
            self.settings.RELOAD_CHECKPOINT_PATH = cpk_path
        self.init()
        self.reload_model_from_cache()
        self.output_path = output_path

    def from_adj_to_graph(self, adj_np, fvs, fvs_outs, all_pos_encs_cache=None, batch_id=None, uid=None):

        G = nx.DiGraph(adj_np)

        g = DGLGraph(G)
        g = dgl.remove_self_loop(g)

        g = g.to('cuda:0')

        g.ndata['fvs'] = fvs
        g.ndata['fvs_out'] = fvs_outs
        all_pos_encs = self.generate_distant_pos_enc(g, batch_id, uid)
        # self.generate_rw_pos_enc(g)
        # self.compute_eigen_basis(g)

        if all_pos_encs_cache is not None:
            all_pos_encs_cache[batch_id] = all_pos_encs
        # if self.trace:
        #     self.distance_sanit_check(g)

        g.add_edges(g.nodes(), g.nodes())
        return g

    def generate_distant_pos_enc(self, g, batch_id, uid):
        n_nodes = g.number_of_nodes()
        # self.collect_common_path(anchors, tree_G, self.settings.POS_ENC_DIM)
        anchors = self.get_anchors_from_cnn_prediction(g, batch_id, uid)
        G = dgl.to_networkx(g.cpu())
        all_pairwise_distances = dict(nx.all_pairs_shortest_path_length(G))
        diameter = nx.algorithms.distance_measures.diameter(G)
        pos_encs = np.empty((n_nodes, len(anchors)), dtype=np.float32)
        all_pos_encs = np.empty((n_nodes, n_nodes), dtype=np.float32)
        for t_node in range(n_nodes):
            dist_dict = all_pairwise_distances[t_node]
            pos_encs[t_node] = np.asarray([dist_dict[a] / float(diameter) for a in anchors])
            all_pos_encs[t_node] = np.asarray([dist_dict[a] / float(diameter) for a in range(n_nodes)])
            # pos_encs[t_node] = np.asarray([1.0 / (dist_dict[a] + 1.0) for a in anchors])
            # all_pos_encs[t_node] = np.asarray([1.0 / (dist_dict[a] + 1.0) for a in range(n_nodes)])

        g.ndata['pos_enc'] = torch.from_numpy(pos_encs).cuda().float()
        g.ndata['p'] = torch.from_numpy(pos_encs).cuda().float()
        return torch.from_numpy(all_pos_encs).cuda()

    def get_anchors_from_cnn_prediction(self, g, batch_id, uid):

        n = g.number_of_nodes()
        fvs_out = F.softmax(g.ndata['fvs_out'], dim=1).cpu().numpy()
        anchors = []
        y_hat = np.zeros((n,), dtype=np.float)
        mask = np.ones((n,), dtype=np.float)

        for label in range(1, 22):
            index = np.argmax(fvs_out[:, label] * mask)
            y_hat[index] = label
            mask[index] = 0.0
            anchors.append(index)
        assert len(y_hat.nonzero()[0]) == 21
        A = g.adjacency_matrix().to_dense().numpy()
        Gvis, adding_anchors = self.add_distal_leafs(anchors[:-3], A)
        anchors = anchors + adding_anchors
        return anchors

    def add_distal_leafs(self, anchors, adj_np):
        upper_tri_adj = np.triu(adj_np)
        G = nx.DiGraph(upper_tri_adj)
        G.remove_edges_from(nx.selfloop_edges(G))
        adding_anchors = []
        for anchor in anchors:
            leafs = {n: nx.shortest_path_length(G, anchor, n)
                     for n in nx.descendants(G, anchor) if G.out_degree(n) == 0}
            if len(leafs) == 0:
                adding_anchors.append(anchor)
            else:
                leafs = sorted(leafs.items(), key=lambda x: x[1])
                adding_anchors.append(leafs[-1][0])
        return G, adding_anchors

    def archive_results(self, test_results):

        predictions, scan_meta = test_results
        # from branch info to skeleton map
        series_uid = scan_meta['uid']
        original_spacing = scan_meta['original_spacing']
        original_size = scan_meta['original_size']
        original_spacing = np.asarray(original_spacing).flatten().tolist()
        origin = np.asarray(scan_meta["origin"]).flatten().tolist()
        direction = np.asarray(scan_meta["direction"]).flatten().tolist()
        assert (predictions.shape == tuple(original_size))
        write_array_to_mhd_itk(self.output_path, [predictions], [series_uid], type=np.uint8,
                               origin=origin[::-1],
                               direction=np.asarray(direction).reshape(3, 3)[
                                         ::-1].flatten().tolist(),
                               spacing=original_spacing[::-1])

    def run(self):
        self.logger.info("Start testing {} scans."
                         .format(len(self.dataset)))
        self.model.eval()
        reversed_relabel_mapping = {v: k for k, v in self.settings.RELABEL_MAPPING.items()}
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.logger.info("Start testing {} scans after exclusion."
                         .format(len(self.dataset)))
        average_time = AverageMeter()
        chunk_size = self.settings.TEST_STITCHES_PATCH_SIZE
        chunk_batch_size = self.settings.TEST_BATCH_SIZE
        try:
            with torch.no_grad():
                for idx, batch_data in enumerate(self.dataset):
                    now = time.time()
                    ref_np = batch_data['#reference']
                    all_labeled_airway_np = batch_data['#all_labeled_reference']
                    branch_info = batch_data['branch_info']
                    scan_meta = batch_data["meta"]
                    series_uid = scan_meta['uid']
                    adj = batch_data['adj']
                    chunked_loader = DataLoader(
                        ChunkCenterBranch({
                            "chunked_list": [torch.from_numpy(all_labeled_airway_np).unsqueeze(0),
                                             torch.from_numpy(all_labeled_airway_np).unsqueeze(0)],
                            "meta": {
                                "uid": [series_uid]
                            }
                        }, chunk_size, [[x] for x in branch_info], all_label_idx=0, ref_idx=-1),
                        batch_size=chunk_batch_size,
                        num_workers=self.settings.NUM_WORKERS,
                        collate_fn=defaut_collate_func, drop_last=False
                    )
                    num_branches = len(branch_info)
                    fvs = torch.zeros((num_branches, self.model.fv_dim))
                    fvs_out = torch.zeros((num_branches, self.settings.NR_CLASS))
                    if torch.cuda.is_available():
                        fvs = fvs.cuda()
                        fvs_out = fvs_out.cuda()
                    all_set_check = set()
                    chunk_centers_list = list(range(num_branches))
                    for cube_idx, chunked_ in enumerate(chunked_loader):
                        r_batch_all, *ignored = chunked_['chunked_list']
                        r_batch_all = r_batch_all.unsqueeze(1)
                        chunk_meta = chunked_['meta']
                        chunk_infos = search_dict_key_recursively(chunk_meta, 'meta', 'info')[0]
                        chunk_ids = [chunk_info[-1] for chunk_info in chunk_infos]
                        chunk_centers = search_dict_key_recursively(chunk_meta, 'meta', 'target_center')[0]
                        r_batch_single_label = self.make_single_labeled_mask(r_batch_all, chunk_ids).float()
                        if self.is_cuda and torch.cuda.is_available():
                            r_batch_single_label = r_batch_single_label.cuda()
                        chunk_feats, chunk_outs = self.model.forward_without_gnn(r_batch_single_label)
                        for cc, cf, cl, co in zip(chunk_centers, chunk_feats,
                                                  chunk_ids, chunk_outs):
                            fvs[cl - 1] = cf.view(-1)
                            fvs_out[cl - 1] = co.view(-1)
                            chunk_centers_list[cl - 1] = cc
                            all_set_check.add(cl - 1)
                    assert (len(list(all_set_check)) == num_branches)
                    g = self.from_adj_to_graph(adj, fvs, fvs_out, uid=series_uid, batch_id=0).to(fvs.device)
                    g = dgl.batch([g])

                    gnn_out, _, _ = self.model.forward(g)
                    gcn_probs = F.softmax(gnn_out, dim=1)
                    pred_gcn_np = self._prediction_by_branch_probs(all_labeled_airway_np, all_labeled_airway_np,
                                                                   chunk_centers_list, gcn_probs)
                    end = time.time()
                    elapse = end - now
                    pred_gcn_np = relabel(pred_gcn_np, reversed_relabel_mapping)
                    original_spacing = scan_meta['original_spacing']
                    original_size = scan_meta['original_size']
                    spacing = scan_meta['spacing']
                    original_spacing = np.asarray(original_spacing).flatten().tolist()
                    original_size = np.asarray(original_size).flatten().tolist()
                    spacing = np.asarray(spacing).flatten().tolist()
                    pred_gcn_np, _ = resample(pred_gcn_np, spacing, factor=2, required_spacing=original_spacing,
                                              new_size=original_size, interpolator='nearest')

                    self.logger.info("VAL Finished scans {}, in {} seconds."
                                     .format(series_uid, elapse))
                    self.logger.info("prediction_np contains: {}".format(np.unique(pred_gcn_np)[1:]))
                    series_archive_path = self.archive_results(
                        (pred_gcn_np,
                         scan_meta))
                    average_time.update(elapse, 1)
                    self.logger.info("Finished {} test scans, archived results in {}, in {} seconds."
                                     .format(idx, series_archive_path, elapse))

        except StopIteration:
            pass
        finally:
            self.logger.info(f"Finished testing, average time = {average_time.avg}")
