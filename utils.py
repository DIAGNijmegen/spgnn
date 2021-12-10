import torch
import logging
from scipy.stats import t
import math
from scipy.sparse import csr_matrix
import numpy as np
import os,csv
from importlib import import_module
from scipy import ndimage
import networkx as nx
import itertools
import copy, operator
from scipy.spatial import distance
import SimpleITK as sitk
from skimage.morphology import skeletonize_3d
import collections

import importlib.util
from pathlib import Path

def convert_dict_string(d, i=1):
    sp = "".join(["    "] * i)
    sp0 = "".join(["    "] * (i - 1))
    s = f"\r\n{sp0}{{"

    for k, v in d.items():
        if isinstance(v, dict):
            s += f"\r\n{sp}{k}:{convert_dict_string(v, i+1)}"
        else:
            s += f"\r\n{sp}{k}:{v}"
    s += f"\r\n{sp0}}}"
    return s

class Settings:
    def __init__(self, settings_module_path, settings_name="settings"):
        # store the settings module in case someone later cares
        self.settings_module_path = settings_module_path
        spec = importlib.util.spec_from_file_location(settings_name, settings_module_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        compulsory_settings = (
            "EXP_NAME",
            "MODEL_NAME",
        )

        self._explicit_settings = set()
        for setting in dir(mod):
            if setting.isupper():
                setting_value = getattr(mod, setting)
                if setting in compulsory_settings and setting is None:
                    raise AttributeError("The %s setting must be Not None. " % setting)
                setattr(self, setting, setting_value)
                self._explicit_settings.add(setting)

    def is_overridden(self, setting):
        return setting in self._explicit_settings

    def __str__(self):
        # return "{}".format(self.__dict__)
        return convert_dict_string(self.__dict__)

def center_skeletion_binary(binary_mask):
    g = make_graph_skeleton(binary_mask > 0)
    c_nodes = nx.center(g)[0]
    g.clear()
    return c_nodes

def merge_dict(list_dict):
    new_d = {}
    for k in list_dict[0].keys():
        new_d[k] = tuple(d[k] for d in list_dict)

    return new_d

def collate_func_nativa(batch):
    merge_d = {}
    for k in batch[0].keys():
        if not isinstance(batch[0][k], (dict, )):
            merge_d[k] = [b[k] for b in batch]
        else:
            merge_d[k] = merge_dict([b[k] for b in batch])

    return merge_d


def calculate_object_labels(preds, targets, check_labels):
    """ preds: D H W, targets: D H W."""
    pred_object_labels = []
    gtd_labels = []
    for label in check_labels:
        t = targets == label
        p = preds == label
        t_sum = np.sum(t)
        p_sum = np.sum(p)
        if t_sum == 0:
            # does not exist in reference
            gtd_labels.append(1)
        else:
            gtd_labels.append(label)  # the label predicted point should have
        if p_sum == 0:
            # does not exist in prediction, we find the what is the label
            # in the prediction in the corresponding area.
            pred_region = preds * t
            if pred_region.sum() == 0:
                pred_object_labels.append(1)
                continue
            ls, l_areas = np.unique(pred_region, return_counts=True)
            assert (len(ls) >= 2)
            if len(ls) > 2:
                pred_label = ls[np.argmax(l_areas[2:]) + 2]
            else:
                pred_label = ls[np.argmax(l_areas[1:]) + 1]
            pred_object_labels.append(pred_label)
            continue
        # compute the label predicted point actually has (largest CC analysis)
        p_cc, n_cc = ndimage.label(p, ndimage.generate_binary_structure(3, 3))
        area_size = np.bincount(p_cc.flat)
        dominant_label = np.argmax(area_size[1:]) + 1
        p_b = p_cc == dominant_label
        p_bs = skeletonize_3d(p_b)
        if np.sum(p_bs) == 0:  # if skeletonize fails
            t_c_nodes = tuple(np.median(np.asarray(np.where(p_b > 0)), 1).astype(np.int16))
        else:
            t_c_nodes = center_skeletion_binary(p_bs > 0)
        pred_object_labels.append(targets[t_c_nodes])
    return gtd_labels, pred_object_labels

def defaut_collate_func(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        # if _use_shared_memory:
        #     # If we're in a background process, concatenate directly into a
        #     # shared memory tensor to avoid an extra copy
        #     numel = sum([x.numel() for x in batch])
        #     storage = batch[0].storage()._new_shared(numel)
        #     out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif isinstance(batch[0], (str, bytes, int, float, tuple)):
        return batch
    elif isinstance(batch[0], list):
        transposed = zip(*batch)
        return [defaut_collate_func(samples) for samples in transposed]
    elif isinstance(batch[0], collections.Mapping):
        return {key: defaut_collate_func([d[key] for d in batch]) for key in batch[0]}
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ == 'ndarray':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    print(batch[0])
    raise TypeError((error_msg.format(type(batch[0]))))

def expand_dims(tensors, expected_dim):
    if tensors.dim() < expected_dim:
        for n in range(expected_dim - tensors.dim()):
            tensors = tensors.unsqueeze(0)

    return tensors

def cumsum(sequence):
    r, s = [], 0
    for e in sequence:
        l = len(e)
        r.append(l + s)
        s += l
    return r

def squeeze_dims(tensors, expected_dim, squeeze_start_index=0):
    if tensors.dim() > expected_dim:
        for n in range(tensors.dim() - expected_dim):
            tensors = tensors.squeeze(squeeze_start_index)

    return tensors


def search_dict_key_recursively(dict_obj, trace_key, find_key):
    find_ = []

    def dict_traverse(dict_obj, trace_key, find_key):
        if not isinstance(dict_obj, dict):
            return
        if find_key in dict_obj.keys():
            find_.append(dict_obj[find_key])
        if trace_key in dict_obj.keys():
            dict_traverse(dict_obj[trace_key], trace_key, find_key)

    dict_traverse(dict_obj, trace_key, find_key)
    return find_

def get_value_recursively(search_dict, field):
    """
    Takes a dict with nested lists and dicts,
    and searches all dicts for a key of the field
    provided.
    """
    fields_found = []

    for key, value in search_dict.items():

        if key == field:
            fields_found.append(value)

        elif isinstance(value, dict):
            results = get_value_recursively(value, field)
            for result in results:
                fields_found.append(result)

        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    more_results = get_value_recursively(item, field)
                    for another_result in more_results:
                        fields_found.append(another_result)

    return fields_found

def make_graph_skeleton(sk_mask):
    """ mask has to be binary and skeletonized already. """
    sk_mask = sk_mask.astype(np.uint8)
    assert (sk_mask.ndim == 3)
    assert sk_mask.sum() > 0
    object_points = np.asarray(np.where(sk_mask > 0)).T.tolist()
    object_points = [tuple(point) for point in object_points]
    g = nx.Graph()
    g.add_nodes_from(object_points)
    # now we add edges by 26 connected neighbors
    for object_point in object_points:
        nei_coords = list(itertools.product(*[(n - 1, n, n + 1) for n in object_point]))
        connected_neibs = [(object_point, nei_coord)
                           for nei_coord in nei_coords if nei_coord in
                           object_points and not nx.has_path(g, object_point, nei_coord)
                           and nei_coord != object_point]
        g.add_edges_from(connected_neibs)

    if not nx.is_connected(g):
        _, cc_num = ndimage.label(sk_mask > 0, ndimage.generate_binary_structure(3, 3))
        sk_num = len(list(nx.connected_components(g)))
        assert (cc_num > 1)
        assert (sk_num == cc_num)
        print("graph is not connected! {} CC ".format(cc_num), flush=True)
        # now we fix this by linking the closest two points from two sub graphs.
        while not nx.is_connected(g):
            end_points = [node for (node, val) in g.degree() if val <= 1]
            cb_ends = list(itertools.combinations(end_points, 2))
            dist_cache = {}
            for cb_end in cb_ends:
                if not nx.has_path(g, cb_end[0], cb_end[1]):
                    d = distance.euclidean(cb_end[0], cb_end[1])
                    dist_cache[cb_end] = d
            sorted_dist_cache = sorted(dist_cache.items(), key=operator.itemgetter(1))
            if len(sorted_dist_cache) > 0:
                cb_s, cb_e = sorted_dist_cache[0][0]
                g.add_edge(cb_s, cb_e)
                print("connect {} to fix.".format((cb_s, cb_e)), flush=True)

    try:
        nx.find_cycle(g, orientation='ignore')
        print("use MST to remove cycles.", flush=True)
        g = nx.minimum_spanning_tree(g)
    except Exception:
        pass
    return g

def relabel(mask, relabel_mapping, skiped_labels=(0, 1, 2), assign_background=True, assign_value=1):
    labels = np.unique(mask)
    labels = [l for l in labels if l not in skiped_labels]
    newmask = copy.deepcopy(mask)
    if assign_background:
        check_labels = set(list(relabel_mapping.keys()) + list(skiped_labels))
        np.putmask(newmask, np.in1d(newmask, list(check_labels), invert=True), assign_value)
    if len(labels) > 1:
        [np.putmask(newmask, mask == label, relabel_mapping[label]) for label in labels if
         label in relabel_mapping.keys()]

    return newmask

def find_shape_outter_boundary(mask, connectivity=1):
    mask = (mask > 0).astype(np.uint8)
    if np.sum(mask) == 0:
        return np.zeros_like(mask)

    template = ndimage.generate_binary_structure(mask.ndim, connectivity)
    # c_mask = ndimage.convolve(mask, template) > 0
    c_mask = ndimage.binary_dilation(mask, template)
    return c_mask & ~mask

def get_stats(array, conf_interval=False, name=None, stdout=False, logout=False):
    """Compute mean and standard deviation from an numerical array

    Args:
        array (array like obj): The numerical array, this array can be
            convert to :obj:`torch.Tensor`.
        conf_interval (bool, optional): If True, compute the confidence interval bound (95%)
            instead of the std value. (default: :obj:`False`)
        name (str, optional): The name of this numerical array, for log usage.
            (default: :obj:`None`)
        stdout (bool, optional): Whether to output result to the terminal.
            (default: :obj:`False`)
        logout (bool, optional): Whether to output result via logging module.
            (default: :obj:`False`)
    """
    eps = 1e-9
    array = torch.Tensor(array)
    std, mean = torch.std_mean(array)
    std = std.item()
    mean = mean.item()
    center = mean

    if conf_interval:
        n = array.size(0)
        se = std / (math.sqrt(n) + eps)
        t_value = t.ppf(0.975, df=n - 1)
        err_bound = t_value * se
    else:
        err_bound = std

    # log and print
    if name is None:
        name = "array {}".format(id(array))
    log = "{}: {:.4f}(+-{:.4f})".format(name, center, err_bound)
    if stdout:
        print(log)
    if logout:
        logging.info(log)

    return center, err_bound


def extract_center_all_labeled(all_label_airway):
    labels = np.unique(all_label_airway)[1:]
    branch_infos = []
    for lid, label in enumerate(labels):
        b_mask = (all_label_airway == label)
        chunk_slices = ndimage.find_objects(b_mask)[0]
        b_chunk = b_mask[chunk_slices]
        sk_chunk = skeletonize_3d(b_chunk)
        sk = np.zeros_like(b_mask)
        sk[chunk_slices] = sk_chunk
        if np.sum(sk) < 3:
            continue
        g = make_graph_skeleton(sk)
        end_points = nx.periphery(g)
        epc = itertools.combinations(end_points, 2)
        path_cache = {}
        for ep in epc:
            d = nx.shortest_path_length(g, ep[0], ep[1])
            path_cache[ep] = d
        sorted_path_cache = sorted(path_cache.items(), key=operator.itemgetter(1))
        if len(sorted_path_cache) == 0:
            continue
        s, e = sorted_path_cache[-1][0]
        path = nx.shortest_path(g, s, e)
        sub_G = nx.Graph()
        sub_G.add_path(path)
        center_node = nx.center(sub_G)[0]
        diameter = nx.diameter(sub_G)

        assert (all_label_airway[center_node] == label)
        branch_infos.append((center_node, diameter, s, e, label))
        # print("processed branch label :{}/{}.".format(lid, len(labels)), flush=True)

    # sort by labels
    *ignored, branch_labels = zip(*branch_infos)
    sorted_ = [branch_infos[i] for i in np.argsort(branch_labels)]
    return sorted_


def read_csv_in_dict(csv_file_path, column_key, fieldnames=None):
    row_dict = {}
    if not os.path.exists(csv_file_path):
        return row_dict, None
    with open(csv_file_path, "rt") as fp:
        cr = csv.DictReader(fp, delimiter=',', fieldnames=fieldnames)
        for row in cr:
            row_dict[row[column_key]] = row

        field_names = cr.fieldnames
    return row_dict, field_names

def get_batch_id(num_nodes: torch.Tensor):
    """Convert the num_nodes array obtained from batch graph to batch_id array
    for each node.
    Args:
        num_nodes (torch.Tensor): The tensor whose element is the number of nodes
            in each graph in the batch graph.
    """
    batch_size = num_nodes.size(0)
    batch_ids = []
    for i in range(batch_size):
        item = torch.full((num_nodes[i],), i, dtype=torch.long, device=num_nodes.device)
        batch_ids.append(item)
    return torch.cat(batch_ids)


def get_indices_sparse(data):
    cols = np.arange(data.size)
    M = csr_matrix((cols, (data.ravel(), cols)), shape=(int(data.max()) + 1, data.size))
    return [np.unravel_index(row.data, data.shape) for row in M]


def topk(x: torch.Tensor, ratio: float, batch_id: torch.Tensor, num_nodes: torch.Tensor):
    """The top-k pooling method. Given a graph batch, this method will pool out some
    nodes from input node feature tensor for each graph according to the given ratio.
    Args:
        x (torch.Tensor): The input node feature batch-tensor to be pooled.
        ratio (float): the pool ratio. For example if :obj:`ratio=0.5` then half of the input
            tensor will be pooled out.
        batch_id (torch.Tensor): The batch_id of each element in the input tensor.
        num_nodes (torch.Tensor): The number of nodes of each graph in batch.

    Returns:
        perm (torch.Tensor): The index in batch to be kept.
        k (torch.Tensor): The remaining number of nodes for each graph.
    """
    batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

    cum_num_nodes = torch.cat(
        [num_nodes.new_zeros(1),
         num_nodes.cumsum(dim=0)[:-1]], dim=0)

    index = torch.arange(batch_id.size(0), dtype=torch.long, device=x.device)
    index = (index - cum_num_nodes[batch_id]) + (batch_id * max_num_nodes)

    dense_x = x.new_full((batch_size * max_num_nodes,), torch.finfo(x.dtype).min)
    dense_x[index] = x
    dense_x = dense_x.view(batch_size, max_num_nodes)

    _, perm = dense_x.sort(dim=-1, descending=True)
    perm = perm + cum_num_nodes.view(-1, 1)
    perm = perm.view(-1)

    k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long) if ratio > 0 else torch.ones_like(num_nodes).long()
    mask = [
        torch.arange(k[i], dtype=torch.long, device=x.device) +
        i * max_num_nodes for i in range(batch_size)]

    mask = torch.cat(mask, dim=0)
    perm = perm[mask]

    return perm, k


def vote_region_based_on_neighbors(mask, voi, connectivity, vote_background=True):
    voi_slices = ndimage.find_objects(voi > 0)
    assert (len(voi_slices) == 1)
    voi_slices = voi_slices[0]
    # need to enlarge it for region of size 1.
    voi_slices = tuple([slice(max(ss.start - 1, 0), min(ss.stop + 1, sp))
                        for ss, sp in zip(voi_slices, mask.shape)])
    voi_r = voi[voi_slices]
    mask_r = mask[voi_slices]
    old_label = np.unique(mask_r[voi_r])
    b_edges = (find_shape_outter_boundary(voi_r, connectivity) * (mask_r > 0)) > 0
    b_edges_labels, b_edges_labels_num = np.unique(mask_r[b_edges], return_counts=True)
    if len(b_edges_labels) == 0 and mask_r[b_edges].sum() == 0:
        if vote_background:
            mask_r[voi_r] = 0
            return old_label, 0
        else:
            return old_label, old_label

    # add_size = (mask == b_edges_labels[np.argmax(b_edges_labels_num)]).sum()
    # old_size = (mask == old_label[0]).sum()
    mask_r[voi_r] = b_edges_labels[np.argmax(b_edges_labels_num)]
    # assert (mask == old_label[0]).sum() == 0
    # assert old_size + add_size == (mask == b_edges_labels[np.argmax(b_edges_labels_num)]).sum()
    return old_label, b_edges_labels[np.argmax(b_edges_labels_num)]

def infer_branch_labels_by_dominance(ref, all_branch_label_map,
                                     return_mapped=True, included_label_index=2):
    labels = np.unique(ref)[included_label_index:]
    mapped = np.zeros_like(ref)
    mapped[ref > 0] = 1
    label_cross_cache = {}
    for label in labels:
        b_ref = ref == label
        wavefront_voi = all_branch_label_map[b_ref]
        w_labels, w_counts = np.unique(wavefront_voi, return_counts=True)
        if 0 in w_labels:
            dominant_label = w_labels[1:][w_counts[1:].argmax()]
            label_cross_cache[dominant_label] = label
        else:
            dominant_label = w_labels[w_counts.argmax()]
            label_cross_cache[dominant_label] = label
        mapped[all_branch_label_map == dominant_label] = label
    if return_mapped:
        return mapped, label_cross_cache
    else:
        return label_cross_cache

def write_array_to_mhd_itk(target_path, arrs, names, type=np.int16,
                           origin=[0.0, 0.0, 0.0],
                           direction=np.eye(3, dtype=np.float64).flatten().tolist(),
                           spacing=[1.0, 1.0, 1.0], orientation='RAI'):
    """ arr is z-y-x, spacing is z-y-x."""
    size = arrs[0].shape
    for arr, name in zip(arrs, names):
        # assert (arr.shape == size)
        simage = sitk.GetImageFromArray(arr.astype(type))
        simage.SetSpacing(np.asarray(spacing, np.float64).tolist())
        simage.SetDirection(direction)
        simage.SetOrigin(origin)
        fw = sitk.ImageFileWriter()
        fw.SetFileName(target_path + '/{}.mhd'.format(name))
        fw.SetDebug(False)
        fw.SetUseCompression(True)
        fw.SetGlobalDefaultDebug(False)
        fw.Execute(simage)
        with open(target_path + '/{}.mhd'.format(name), "rt") as fp:
            lines = fp.readlines()
            for idx, line in enumerate(lines):
                if "AnatomicalOrientation" in line:
                    header = line[:line.find("=")].strip()
                    newline = "{}={}".format(header, orientation) + os.linesep
                    lines[idx] = newline
                    break
        with open(target_path + '/{}.mhd'.format(name), "wt", newline='') as fp:
            fp.writelines(lines)

def get_callable_by_name(module_name):
    cls = getattr(import_module(module_name.rpartition('.')[0]),
                  module_name.rpartition('.')[-1])
    return cls

def resample_sitk_image(sitk_image, spacing=None, interpolator=None,
                        fill_value=0, new_size=None):
    """Resamples an ITK image to a new grid. If no spacing is given,
    the resampling is done isotropically to the smallest value in the current
    spacing. This is usually the in-plane resolution. If not given, the
    interpolation is derived from the input data type. Binary input
    (e.g., masks) are resampled with nearest neighbors, otherwise linear
    interpolation is chosen.
    Parameters
    ----------
    sitk_image : SimpleITK image or str
      Either a SimpleITK image or a path to a SimpleITK readable file.
    spacing : tuple
      Tuple of integers
    interpolator : str
      Either `nearest`, `linear` or None.
    fill_value : int
    Returns
    -------
    SimpleITK image.
    """
    _SITK_INTERPOLATOR_DICT = {
        'nearest': sitk.sitkNearestNeighbor,
        'linear': sitk.sitkLinear,
        'gaussian': sitk.sitkGaussian,
        'label_gaussian': sitk.sitkLabelGaussian,
        'bspline': sitk.sitkBSpline,
        'hamming_sinc': sitk.sitkHammingWindowedSinc,
        'cosine_windowed_sinc': sitk.sitkCosineWindowedSinc,
        'welch_windowed_sinc': sitk.sitkWelchWindowedSinc,
        'lanczos_windowed_sinc': sitk.sitkLanczosWindowedSinc
    }
    if isinstance(sitk_image, str):
        sitk_image = sitk.ReadImage(sitk_image)
    num_dim = sitk_image.GetDimension()

    if not interpolator:
        interpolator = 'linear'
        pixelid = sitk_image.GetPixelIDValue()

        if pixelid not in [1, 2, 4]:
            raise NotImplementedError(
                'Set `interpolator` manually, '
                'can only infer for 8-bit unsigned or 16, '
                '32-bit signed integers')
        if pixelid == 1:  # 8-bit unsigned int
            interpolator = 'nearest'

    orig_pixelid = sitk_image.GetPixelIDValue()
    orig_origin = sitk_image.GetOrigin()
    orig_direction = sitk_image.GetDirection()
    orig_spacing = np.array(sitk_image.GetSpacing())
    orig_size = np.array(sitk_image.GetSize(), dtype=np.int)

    if not spacing:
        min_spacing = orig_spacing.min()
        new_spacing = [min_spacing] * num_dim
    else:
        new_spacing = [float(s) for s in spacing]

    assert interpolator in _SITK_INTERPOLATOR_DICT.keys(), \
        '`interpolator` should be one of {}'.format(
            _SITK_INTERPOLATOR_DICT.keys())

    sitk_interpolator = _SITK_INTERPOLATOR_DICT[interpolator]

    if new_size is None:
        new_size = orig_size * (orig_spacing / new_spacing)
        new_size = np.ceil(new_size).astype(
            np.int)  # Image dimensions are in integers
        new_size = [int(s) for s in
                    new_size]  # SimpleITK expects lists, not ndarrays

    resample_filter = sitk.ResampleImageFilter()

    resampled_sitk_image = resample_filter.Execute(sitk_image,
                                                   new_size,
                                                   sitk.Transform(),
                                                   sitk_interpolator,
                                                   orig_origin,
                                                   new_spacing,
                                                   orig_direction,
                                                   fill_value,
                                                   orig_pixelid)

    return resampled_sitk_image

def resample(narray, orig_spacing, factor=2, required_spacing=None, new_size=None, interpolator='linear'):
    if new_size is not None and narray.shape == new_size:
        print("size is equal not resampling!")
        return narray, orig_spacing
    s_image = sitk.GetImageFromArray(narray)
    s_image.SetSpacing(np.asarray(orig_spacing[::-1], dtype=np.float64).tolist())

    req_spacing = factor * np.asarray(orig_spacing)
    req_spacing = tuple([float(s) for s in req_spacing])
    if required_spacing is not None:
        req_spacing = required_spacing
    if new_size:
        new_size = new_size[::-1]
    resampled_image = resample_sitk_image(s_image,
                                          spacing=req_spacing[::-1],
                                          interpolator=interpolator,
                                          fill_value=0, new_size=new_size)

    resampled = sitk.GetArrayFromImage(resampled_image)

    return resampled, req_spacing

class Resample(object):
    """Convert ndarrays in sample to Tensors if # sign and "image" tag in its keys."""

    def __init__(self, mode, factor, size=None):
        self.mode = mode
        self.factor = factor
        if size:
            self.size = list(size)

    def __call__(self, sample):
        new_sample = {"meta": {}}
        spacing = sample['meta']['spacing']
        if self.mode == 'random_spacing':
            factor = np.random.uniform(self.factor[0], self.factor[1])
            require_spacing = [factor] * len(spacing)
            new_size = None
        elif self.mode == 'fixed_factor':
            factor = self.factor
            require_spacing = None
            new_size = None
        elif self.mode == 'fixed_spacing':
            if isinstance(self.factor, (float, int)):
                factor = self.factor
                require_spacing = [factor] * len(spacing)
            elif isinstance(self.factor, (tuple, list)):
                require_spacing = self.factor
                factor = 2  # dummy number meaningless.
            new_size = None
        elif self.mode == "inplane_spacing_only":
            current_size = sample['meta']['size']
            assert (len(current_size) == 3)
            require_spacing = [spacing[0], self.factor[1],
                               self.factor[2]]
            new_size = None
            factor = 2
        elif self.mode == "inplane_resolution_only":
            current_size = sample['meta']['size']
            assert (len(current_size) == 3)
            require_spacing = [spacing[0], spacing[1] * current_size[1] / self.size[1],
                               spacing[2] * current_size[2] / self.size[2]]
            new_size = [current_size[0], self.size[1],
                        self.size[2]]
            factor = 2
        elif self.mode == "inplane_resolution_z_spacing":
            current_size = sample['meta']['size']
            assert (len(current_size) == 3)
            require_spacing = [self.factor[0], spacing[1] * current_size[1] / self.size[1],
                               spacing[2] * current_size[2] / self.size[2]]
            new_size = [int(round(current_size[0] * spacing[0] / self.factor[0])), self.size[1],
                        self.size[2]]
            factor = 2
        elif self.mode == "inplane_resolution_z_jittering":
            current_size = sample['meta']['size']
            assert (len(current_size) == 3)
            z_spacing_base = spacing[0]
            offset = np.random.uniform(-self.factor, self.factor)
            z_spacing = z_spacing_base + offset
            require_spacing = [z_spacing, spacing[1] * current_size[1] / self.size[1],
                               spacing[2] * current_size[2] / self.size[2]]
            new_size = [int(round(current_size[0] * spacing[0] / z_spacing)), self.size[1],
                        self.size[2]]
            factor = 2
        elif self.mode == "inplane_resolution_min_z_spacing":
            current_size = sample['meta']['size']
            assert (len(current_size) == 3)
            if spacing[0] < self.factor[0]:
                print("set spacing to {} from {}.".format(self.factor[0], spacing[0]))
                require_spacing = [self.factor[0], spacing[1] * current_size[1] / self.size[1],
                                   spacing[2] * current_size[2] / self.size[2]]
                new_size = [int(round(current_size[0] * spacing[0] / self.factor[0])), self.size[1],
                            self.size[2]]
            else:
                require_spacing = [spacing[0], spacing[1] * current_size[1] / self.size[1],
                                   spacing[2] * current_size[2] / self.size[2]]
                new_size = [current_size[0], self.size[1],
                            self.size[2]]
            factor = 2
        elif self.mode == "fixed_spacing_min_in_plane_resolution":
            current_size = sample['meta']['size']
            assert (len(current_size) == 3)
            if not isinstance(self.factor, (tuple, list)):
                factor = [self.factor] * 3
            else:
                factor = self.factor
            new_y_size = int(round(current_size[1] * spacing[1] / factor[1]))
            if new_y_size > self.size[1]:

                require_spacing = [spacing[0], spacing[1] * current_size[1] / self.size[1],
                                   spacing[2] * current_size[2] / self.size[2]]
                new_size = [current_size[0], self.size[1],
                            self.size[2]]
                print(
                    "new_size:{} > target_size {}. fixed_in_plane_resolution mode. {}.".format(new_y_size, self.size[1],
                                                                                               new_size))
            else:
                require_spacing = [spacing[0], factor[1],
                                   factor[2]]
                new_size = None
                print(
                    "new_size:{} <= target_size {}. fixed_spacing. {}.".format(new_y_size, self.size[1],
                                                                               require_spacing))
            factor = 2
        elif self.mode == "iso_minimal":
            factor = spacing[0]
            require_spacing = [np.min(spacing)] * len(spacing)
            new_size = None
        elif self.mode == "fixed_output_size":
            current_size = sample['meta']['size']
            ratio = current_size[-1] / self.size[-1]
            require_spacing = [spacing[-1] * ratio] * len(spacing)
            new_size = self.size[:]
            new_size[0] = int(round(current_size[0] * spacing[0] / require_spacing[0]))
            new_size[1] = int(round(current_size[1] * spacing[1] / require_spacing[1]))
            factor = 2
        elif self.mode == "fixed_size":
            current_size = sample['meta']['size']
            ratios = np.asarray(current_size) / np.asarray(self.size)
            require_spacing = (spacing * ratios).tolist()
            new_size = self.size[:]
            factor = 2
        elif self.mode == "spacing_size_match":
            require_spacing = self.factor[:]
            new_size = self.size[:]
            factor = 2
        else:
            raise NotImplementedError
        for k, v in sample.items():
            if "#" in k:
                if "reference" in k or 'weight_map' in k:
                    mode = 'nearest'
                else:
                    mode = 'linear'
                if v.ndim == 4:
                    r_results = [resample(vv, spacing, factor=factor,
                                          required_spacing=require_spacing,
                                          new_size=new_size, interpolator=mode) for vv
                                 in v]
                    new_spacing = r_results[0][-1]
                    nv = np.stack([r[0] for r in r_results], axis=0)
                elif v.ndim == 3:
                    nv, new_spacing = resample(v, spacing, factor=factor,
                                               required_spacing=require_spacing,
                                               new_size=new_size, interpolator=mode)
                else:
                    raise NotImplementedError
                new_sample[k] = nv
                new_size = nv.shape
            else:
                new_sample[k] = v
        old_size = sample['meta']['size']
        new_sample['meta'] = copy.deepcopy(sample['meta'])
        new_sample['meta']['spacing'] = tuple(new_spacing)
        new_sample['meta']['size'] = new_size
        new_sample['meta']['size_before_resample'] = old_size
        new_sample['meta']['resample_factor'] = factor
        return new_sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors if # sign in its keys."""

    def __call__(self, sample, is_pin=False):
        if is_pin:
            sample = {(k if "#" in k else k): (torch.from_numpy(v.copy()).pin_memory() if "#" in k else v)
                      for k, v in sample.items()}
        else:
            sample = {(k if "#" in k else k): (torch.from_numpy(v.copy()) if "#" in k else v)
                      for k, v in sample.items()}
        return sample

class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count