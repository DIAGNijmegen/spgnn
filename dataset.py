from torch.utils.data import Dataset
from utils import *
import random, os
import numpy as np
import torch
import traceback
import SimpleITK as sitk
from scipy import ndimage, spatial
import torch.nn.functional as F
import bisect
import operator
import networkx as nx
from skimage.morphology import skeletonize_3d, remove_small_objects
from scipy import ndimage
import shutil
import re
from collections import defaultdict
from networkx.drawing.nx_pydot import graphviz_layout
import functools, glob
from pathlib import Path
import pickle


class ConvEmbeddingDataset(Dataset):

    @classmethod
    def get_series_uids(cls, csv_file):
        scan_selected, _ = read_csv_in_dict(csv_file, 'SeriesInstanceUID')
        return sorted(list(scan_selected.keys()))

    def __init__(self, archive_path, series_uids, transforms=None, keep_sorted=True):
        super(ConvEmbeddingDataset, self).__init__()
        self.keep_sorted = keep_sorted
        if not self.keep_sorted:
            self.series_uids = random.sample(series_uids, len(series_uids))
        else:
            self.series_uids = series_uids
        self.uid_indice_map = {uid: idx for idx, uid in enumerate(self.series_uids)}
        self.fe_path = archive_path + '/derived/conv_embedding/'
        self.transforms = transforms

    def __len__(self):
        return len(self.series_uids)

    def __getitem__(self, scan_index):
        series_uid = self.series_uids[scan_index]
        # step 1, find all metas
        fe = np.load(os.path.join(self.fe_path, f"{series_uid}.pkl"), allow_pickle=True)
        return fe


class AirwayTreeGraphDataset(Dataset):

    @classmethod
    def get_series_uids(cls, csv_file):
        scan_selected, _ = read_csv_in_dict(csv_file, 'SeriesInstanceUID')
        return sorted(list(scan_selected.keys()))

    def __init__(self, archive_path, series_uids, transforms=None, keep_sorted=True):
        super(AirwayTreeGraphDataset, self).__init__()
        self.keep_sorted = keep_sorted
        if not self.keep_sorted:
            self.series_uids = random.sample(series_uids, len(series_uids))
        else:
            self.series_uids = series_uids
        self.uid_indice_map = {uid: idx for idx, uid in enumerate(self.series_uids)}
        self.fb_path = archive_path + '/derived/conv/'
        self.transforms = transforms

    def __len__(self):
        return len(self.series_uids)

    def __getitem__(self, scan_index):
        series_uid = self.series_uids[scan_index]
        with open(os.path.join(self.fb_path, f"{series_uid}.pkl"), 'rb') as fp:
            d = pickle.load(fp)

        return ToTensor()(d)


class ChunkCenterBranch(Dataset):

    def __init__(self, chunked, resolutions, branch_infos, all_label_idx, ref_idx=-1,
                 transforms=None):
        super(ChunkCenterBranch, self).__init__()
        self.tensors_list = chunked["chunked_list"]
        # tensors are in batch first dim ordering.
        self.meta = chunked["meta"]
        self.resolutions = resolutions
        self.transforms = transforms
        self.ref_idx = ref_idx
        self.all_label_idx = all_label_idx
        self.branch_infos = branch_infos

        self.window_locations_type = []
        self.window_locations_list = []
        self.window_locations_info = []
        t = self.tensors_list[self.ref_idx]
        for _ in t:
            centers = []
            types = []
            infos = []
            for branch_info in self.branch_infos:
                center_node, diameter, anatomical_label, l_start, l_cursor, bid = branch_info[0]
                assert (anatomical_label >= 1)
                centers.append(center_node)
                types.append(anatomical_label)
                infos.append((center_node, diameter, anatomical_label, l_start, l_cursor, bid))
            print("extracted {} airway branches, with {}.".format(len(self.branch_infos),
                                                                  np.unique(types, return_counts=True)))
            self.window_locations_list.append(centers)
            self.window_locations_type.append(types)
            self.window_locations_info.append(infos)

    def __len__(self):
        return sum([len(window_locations) for window_locations in self.window_locations_list])

    def __getitem__(self, idx):
        # the window_location_list is mutable and may be modified by augmenters. therefore,
        # we need to dynamically calculate indices.
        cumulative_sizes = cumsum(self.window_locations_list)
        tensor_idx = bisect.bisect_right(cumulative_sizes, idx)
        if tensor_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - cumulative_sizes[tensor_idx - 1]

        chunk_tensor_list = []
        pad_size_list = []
        resolution = self.resolutions[self.ref_idx]
        target_center = self.window_locations_list[tensor_idx][sample_idx]
        target_type = int(self.window_locations_type[tensor_idx][sample_idx])
        target_info = self.window_locations_info[tensor_idx][sample_idx]
        for tlid, tensors in enumerate(self.tensors_list):
            current_t = tensors[tensor_idx]
            slices = tuple([slice(max(0, tc - rs // 2), min(tc + rs - rs // 2, sp))
                            for tc, rs, sp in zip(target_center, resolution,
                                                  current_t.shape)])
            pad_size = tuple([(rs // 2 - tc if tc - rs // 2 < 0 else 0,
                               tc + rs - rs // 2 - sp if tc + rs - rs // 2 > sp else 0)
                              for tc, rs, sp in zip(target_center, resolution,
                                                    current_t.shape)])
            pad_size_list.append(pad_size)
            chunk_tensor = current_t[slices]
            chunk_tensor = F.pad(chunk_tensor, tuple(np.asarray(pad_size[::-1]).flatten().tolist()),
                                 "constant", 0)
            assert (chunk_tensor.shape == resolution)
            chunk_tensor_list.append(chunk_tensor)
            if tlid == self.ref_idx:
                assert (target_type == int(current_t[target_center]))
            if tlid == self.all_label_idx:
                assert (target_info[-1] == int(current_t[target_center].item()))

        def f(a):
            if isinstance(a, dict):
                return dict(zip(a, map(f, a.values())))
            else:
                return a[tensor_idx]

        old_dict = dict(zip(self.meta, map(f, self.meta.values())))
        ret = {
            "chunked_list": chunk_tensor_list,
            "meta": {
                "tensor_idx": tensor_idx,
                "sample_idx": sample_idx,
                "target_center": tuple(target_center),
                "info": tuple(target_info),
                "type": target_type,
                "meta": old_dict
            }
        }
        if self.transforms:
            ret = self.transforms(ret)
        return ret


class COPDGeneChunkAirway18Labels(Dataset):
    meta_subjects_name = 'meta_subjects.csv'
    meta_scans_name = 'meta_scans.csv'
    ON_PREMISE_ROOT = None

    @classmethod
    def get_series_uids(cls, csv_file):
        scan_selected, _ = read_csv_in_dict(csv_file, 'SeriesInstanceUID')
        return sorted(list(scan_selected.keys()))

    def __len__(self):
        return len(self.series_uids)

    def __getitem__(self, scan_index):
        series_uid = self.series_uids[scan_index]
        # ret = self.get_data(series_uid)
        try:
            ret = self.get_data(series_uid)
            ret.update({"uid": series_uid})
        except Exception as e:
            print("Exception occurs : {}".format(e))
            traceback.print_exc()
            traceback.print_stack()
            ret = {"uid": series_uid}
        return ret

    def get_all_labeled_tree(self, series_uid):
        all_labeled_path = self.all_label_reference_folder + "/{}.mhd".format(series_uid)
        all_labeled_zraw_path = self.all_label_reference_folder + "/{}.zraw".format(series_uid)
        if self.ON_PREMISE_ROOT is not None:
            ref_path = self.ON_PREMISE_ROOT + '/all_labeled/'
            if not os.path.exists(ref_path):
                os.makedirs(ref_path)
            on_premise_path = os.path.join(ref_path, "{}.mhd".format(series_uid))
            on_premise_zraw_path = os.path.join(ref_path, "{}.zraw".format(series_uid))
            try:
                if not os.path.exists(on_premise_path):
                    print("copying file to on premise path {}.".format(on_premise_path))
                    on_premise_path = shutil.copyfile(all_labeled_path, on_premise_path)
                if not os.path.exists(on_premise_zraw_path):
                    print("copying file to on premise path {}.".format(on_premise_zraw_path))
                    shutil.copyfile(all_labeled_zraw_path, on_premise_zraw_path)
                f_ = sitk.ReadImage(on_premise_path)
            except Exception as e:
                print(
                    "loading file or copying  at {} failed with {},"
                    " now read from {}.".format(on_premise_path, e, all_labeled_path))
                f_ = sitk.ReadImage(all_labeled_path)
            return f_
        return sitk.ReadImage(all_labeled_path)

    def __init__(self, archive_path, series_uids, keep_sorted=True, trace_path=None):
        super(COPDGeneChunkAirway18Labels, self).__init__()
        self.keep_sorted = keep_sorted
        self.trace_path = trace_path
        if not self.keep_sorted:
            self.series_uids = random.sample(series_uids, len(series_uids))
        else:
            self.series_uids = series_uids
        self.archive_path = archive_path
        # read all COPDGene metadata of scans/subjects.
        meta_subjects_file_path = os.path.join(archive_path, self.meta_subjects_name)
        meta_scans_file_path = os.path.join(archive_path, self.meta_scans_name)
        self.meta_subjects, self.fd_md_subjects = read_csv_in_dict(meta_subjects_file_path, 'sid')
        self.meta_scans, self.fd_md_scans = read_csv_in_dict(meta_scans_file_path, 'SeriesInstanceUID')

        self.reference_folder = archive_path + '/derived/seg-airways-chunk/'
        self.all_label_reference_folder = archive_path + '/derived/seg-airways-chunk-labeled/'
        # verify that series_uids exists in reference_folder
        for series_uid in series_uids:
            reference_folders = [self.reference_folder + '/{}.mhd'.format(series_uid),
                                 self.reference_folder + '/{}.zraw'.format(series_uid)]
            if any(not os.path.exists(x) for x in reference_folders):
                raise AttributeError("{} is not in reference_folder.".format(series_uid))

    def get_reference(self, series_uid):
        chunk_scan_path = self.reference_folder + "/{}.mhd".format(series_uid)
        chunk_scan_zraw_path = self.reference_folder + "/{}.zraw".format(series_uid)
        if self.ON_PREMISE_ROOT is not None:
            ref_path = self.ON_PREMISE_ROOT + '/ref/'
            if not os.path.exists(ref_path):
                os.makedirs(ref_path)
            on_premise_path = os.path.join(ref_path, "{}.mhd".format(series_uid))
            on_premise_zraw_path = os.path.join(ref_path, "{}.zraw".format(series_uid))
            try:
                if not os.path.exists(on_premise_path):
                    print("copying file to on premise path {}.".format(on_premise_path))
                    on_premise_path = shutil.copyfile(chunk_scan_path, on_premise_path)
                if not os.path.exists(on_premise_zraw_path):
                    print("copying file to on premise path {}.".format(on_premise_zraw_path))
                    shutil.copyfile(chunk_scan_zraw_path, on_premise_zraw_path)
                f_ = sitk.ReadImage(on_premise_path)
            except Exception as e:
                print(
                    "loading file or copying  at {} failed with {},"
                    " now read from {}.".format(on_premise_path, e, chunk_scan_path))
                f_ = sitk.ReadImage(chunk_scan_path)
            return f_
        return sitk.ReadImage(chunk_scan_path)

    def visualize_airway_graph(self, uid, graph, labels_mappings):
        import matplotlib.pyplot as plt
        all_label_mapping = {n: 1 for n in graph.nodes()}
        all_label_mapping.update({k - 1: v for k, v in labels_mappings.items()})
        plt.clf()
        pos = graphviz_layout(graph, prog='dot', root=0)
        # pos_dict = {n:graph.nodes[n]['point'][:-1] for n in graph.nodes()}
        # l = nx.spring_layout(graph, dim=2, pos=pos_dict, scale=2.0)
        nx.draw(graph, pos, labels=all_label_mapping, with_labels=True, arrows=False, node_size=150, font_size=10)
        plt.savefig('{}/{}.png'.format(self.trace_path, uid))


    def transform_adj(self, adj):
        return adj

    def build_wave_front_tree(self, all_label_mask):
        # use only the largest connection component
        struct_e = ndimage.generate_binary_structure(3, 2)
        labeled, cc_num = ndimage.label(all_label_mask > 0, struct_e)
        max_id = np.bincount(labeled.flat)[1:].argmax()
        largest_cc = labeled == (max_id + 1)
        all_label_mask = all_label_mask * largest_cc
        # get rid of small labels.
        old_labels = np.unique(all_label_mask)[1:]
        r_all_label_mask = remove_small_objects(all_label_mask, min_size=5, in_place=False)
        cut_region = np.logical_xor(r_all_label_mask > 0, all_label_mask > 0)

        label_cut_region, cc_num = ndimage.label(cut_region, struct_e)
        for label in range(1, cc_num + 1):
            cut_b = label_cut_region == label
            ol, nl = vote_region_based_on_neighbors(all_label_mask, cut_b, 2)
            print("vote cut region {} -> {}.".format(ol, nl), flush=True)

        # sanity check
        r_all_label_mask = remove_small_objects(all_label_mask, min_size=5, in_place=False)
        cut_region = np.logical_xor(r_all_label_mask > 0, all_label_mask > 0)
        assert cut_region.sum() == 0
        labels = np.unique(all_label_mask)[1:]

        print("remove small labels : {} objects -> {}.".format(len(old_labels), len(labels)),
              flush=True)
        # for each label, get rid of small cc.
        object_slices = ndimage.find_objects(all_label_mask)
        for idx, object_slice in enumerate(object_slices):
            if object_slice == None:
                continue
            label = idx + 1
            al_chunk = all_label_mask[object_slice]
            al_chunk_b = al_chunk == label
            al_chunk_labeled, cc_num = ndimage.label(al_chunk_b, struct_e)
            if cc_num != 1:
                _lb, _ls = np.unique(al_chunk_labeled, return_counts=True)
                print("bad cc occurs: {} label has {} cc {} of sizes {}.".format(label, cc_num,
                                                                                 _lb[1:], _ls[1:]), flush=True)

                for i in np.argsort(_ls[1:])[:-1]:
                    bbi = al_chunk_labeled == _lb[i + 1]
                    ol, nl = vote_region_based_on_neighbors(al_chunk, bbi, 2)
                    print("vote smaller cc {} -> {}.".format(ol, nl), flush=True)

        # relabel the label map so the labels are sorted, starting at 1.
        labels = np.unique(all_label_mask)[1:]
        relabeled_all_label = np.zeros_like(all_label_mask, np.int)
        for idx, label in enumerate(sorted(labels)):
            relabeled_all_label[all_label_mask == label] = idx + 1
        all_label_mask = relabeled_all_label
        labels = np.unique(all_label_mask)[1:]
        print("relabled {} labels here.".format(len(labels)), flush=True)
        # step.2 now we build tree
        coordinates = np.asarray(np.where(all_label_mask > 0)).T
        kd_tree = spatial.cKDTree(coordinates)
        knn_dist, knn_cds = kd_tree.query(coordinates, k=26, distance_upper_bound=2)
        knn_cds_dict = {tuple(c): tuple(coordinates[e[np.nonzero(kdst != np.inf)]].T)
                        for c, kdst, e in zip(coordinates, knn_dist, knn_cds)}
        all_label_slices = ndimage.find_objects(all_label_mask)
        assert len(list(all_label_slices)) == len(labels)

        label_center_cache = {}
        g = nx.Graph()
        v_adj = np.zeros((len(labels), len(labels)))
        for idx, all_label_slice in enumerate(all_label_slices):
            label = idx + 1
            b_mask = all_label_mask == label
            b_cds = np.asarray(np.where(b_mask > 0)).T
            b_slices = tuple([slice(max(0, ss.start - 3), min(ss.stop + 3, sp))
                              for ss, sp in zip(all_label_slice, all_label_mask.shape)])
            b_mask_chunk = b_mask[b_slices]
            _, cc_num = ndimage.label(b_mask_chunk, struct_e)
            assert cc_num == 1
            b_mask_sk = skeletonize_3d(b_mask_chunk)
            if np.sum(b_mask_sk) == 0:
                b_mask_chunk_coor = np.asarray(np.where(b_mask_chunk > 0))
                tls, brs = np.min(b_mask_chunk_coor, 1), np.max(b_mask_chunk_coor, 1)
                diameter = float(np.max([br - tl for tl, br in zip(tls, brs)]))
                b_mask_center = tuple(np.mean(b_mask_chunk_coor, 1).astype(np.int16))
                if not b_mask_chunk[b_mask_center] > 0:
                    b_mask_chunk_dist = ndimage.distance_transform_edt(b_mask_chunk)
                    b_mask_center = np.unravel_index(np.argmax(b_mask_chunk_dist, axis=None), b_mask_chunk_dist.shape)
                    assert b_mask_chunk[b_mask_center] > 0
            else:
                sub_g = make_graph_skeleton(b_mask_sk)
                e_sub_g = nx.eccentricity(sub_g)
                diameter = max(e_sub_g.values())
                b_mask_center = nx.center(sub_g)[0]
                sub_g.clear()
            # b_mask_dist = ndimage.distance_transform_edt(b_mask[b_slices])
            # b_mask_center = np.unravel_index(np.argmax(b_mask_dist, axis=None), b_mask_dist.shape)
            top_left = [ss.start for ss in b_slices]
            b_mask_center = tuple((np.asarray(b_mask_center) + top_left).tolist())
            label_center_cache[label] = b_mask_center
            assert all_label_mask[b_mask_center] == label
            # query node labels in binary mask.
            b_knn = [all_label_mask[knn_cds_dict[tuple(b_cd)]].tolist() for b_cd in b_cds]
            neighbor_labels = list(set(np.unique([xx for x in b_knn for xx in x])) - {0, label})
            # print("neighbor_labels of {}: {}".format(label, neighbor_labels), flush=True)
            assert len(neighbor_labels) > 0
            # assert len(neighbor_labels) <= 3 and len(neighbor_labels) >= 1
            g.add_node(label - 1, point=b_mask_center, length=diameter)
            for nl in neighbor_labels:
                if nl in label_center_cache.keys():
                    g.add_edge(label - 1, nl - 1,
                               distance=spatial.distance.euclidean(
                                   label_center_cache[nl],
                                   label_center_cache[label])
                               )
                else:
                    g.add_edge(label - 1, nl - 1)
                v_adj[label - 1, nl - 1] = 1
                v_adj[nl - 1, label - 1] = 1
            v_adj[label - 1, label - 1] = 1

        assert g.number_of_nodes() == len(labels)
        assert nx.is_connected(g)
        try:
            cycles = nx.find_cycle(g, orientation='ignore')
            print("found {} cycles.".format(cycles), flush=True)
            g = nx.minimum_spanning_tree(g)
        except Exception:
            print("no cycle is found!", flush=True)

        assert nx.is_tree(g)
        adj = nx.to_numpy_array(g, nodelist=list(range(len(labels))))
        adj = np.eye(adj.shape[0]) + adj
        print("ADJ check sum: {}".format((v_adj - adj).sum()), flush=True)
        # build branch record list
        branch_records = [(g.nodes[n]['point'], g.nodes[n]['length'],
                           g.nodes[n]['point'], g.nodes[n]['point'], n + 1) for n in sorted(g.nodes())]

        adj = self.transform_adj(adj)
        return adj, g, branch_records, all_label_mask

    def label_main_brochial(self, wavefront_ref_map, adj):
        reverse_wavefront_ref_map = {v: k for k, v in wavefront_ref_map.items()}
        adj_no_self = adj - np.eye(adj.shape[0])
        adj_uh = np.triu(adj_no_self)
        d_g = nx.DiGraph(adj_uh)
        RMB_label = nx.shortest_path(d_g, 0, reverse_wavefront_ref_map[2] - 1)[1] + 1
        assert (RMB_label == nx.shortest_path(d_g, 0, reverse_wavefront_ref_map[3] - 1)[1] + 1)

        LMB_label = nx.shortest_path(d_g, 0, reverse_wavefront_ref_map[13] - 1)[1] + 1
        assert (LMB_label == nx.shortest_path(d_g, 0, reverse_wavefront_ref_map[14] - 1)[1] + 1)
        return RMB_label, LMB_label

    def get_data(self, series_uid):

        seg_airway_chunk = self.get_reference(series_uid)
        seg_airway_chunk_labeled = self.get_all_labeled_tree(series_uid)
        assert (seg_airway_chunk.GetSize()
                == seg_airway_chunk_labeled.GetSize())

        base_dict = {
            'uid': series_uid,
        }
        base_dict.update({"size": seg_airway_chunk.GetSize()[::-1],
                          "spacing": seg_airway_chunk.GetSpacing()[::-1],
                          "original_spacing": seg_airway_chunk.GetSpacing()[::-1],
                          "original_size": seg_airway_chunk.GetSize()[::-1],
                          "origin": seg_airway_chunk.GetOrigin()[::-1],
                          "direction": np.asarray(seg_airway_chunk.GetDirection()).reshape(3, 3)[
                                       ::-1].flatten().tolist()})
        airway = sitk.GetArrayFromImage(seg_airway_chunk).astype(np.uint8)
        labels = np.unique(airway)
        airway_known = np.in1d(airway.flat, [0, 1, 2, 3, 4, 5, 6, 7, 8,
                                             9, 10, 11, 12, 14, 15, 16, 17, 18, 20, 21])
        airway.flat[~airway_known] = 1
        print("{} contains labels {}.".format(series_uid, labels))
        RELABEL_MAPPING = {14: 13, 15: 14, 16: 15, 17: 16, 18: 17, 20: 18, 21: 19}
        airway = relabel(airway, RELABEL_MAPPING, assign_background=False)
        all_label_airway = sitk.GetArrayFromImage(seg_airway_chunk_labeled).astype(np.int16)
        all_label_airway = (all_label_airway * (airway > 0)).astype(np.int16)
        ret = {
            "#reference": airway,
            "#all_labeled_reference": all_label_airway,
            "meta": base_dict,
        }

        ret = Resample('fixed_spacing', (0.625, 0.5, 0.5))(ret)
        # here to compute branch infos and adjacent matrix

        airway = ret['#reference']
        all_label_airway = ret['#all_labeled_reference']
        adj, graph, branch_infos, relabel_all_label_airway = self.build_wave_front_tree(all_label_airway)
        wavefront_ref_map = infer_branch_labels_by_dominance(airway, relabel_all_label_airway,
                                                             return_mapped=False)
        rmb, lmb = self.label_main_brochial(wavefront_ref_map, adj)
        airway[relabel_all_label_airway == 1] = 20
        airway[relabel_all_label_airway == rmb] = 21
        airway[relabel_all_label_airway == lmb] = 22
        wavefront_ref_map = infer_branch_labels_by_dominance(airway, relabel_all_label_airway,
                                                             return_mapped=False)

        if self.trace_path is not None:
            self.visualize_airway_graph(series_uid, graph, wavefront_ref_map)
            self.visualize_labeled_tree(series_uid, airway, relabel_all_label_airway, wavefront_ref_map)
        # sanity check
        new_branch_infos = []
        for idx, branch_info in enumerate(branch_infos):
            center, diameter, start, end, branch_id = branch_info
            if airway[center] not in list(range(1, 23)):
                print("{}, {}.".format(airway[center], branch_id))
            assert (airway[center] in list(range(1, 23)))
            assert (relabel_all_label_airway[center] == int(branch_id))
            assert ((idx + 1) == int(branch_id))
            if int(branch_id) in wavefront_ref_map.keys():
                anatomical_label = wavefront_ref_map[int(branch_id)]
                if anatomical_label != airway[center]:
                    print("dominant label {} is not consistent with center label {}."
                          .format(anatomical_label, airway[center]), flush=True)
            else:
                anatomical_label = 1

            new_branch_infos.append((center, diameter, int(anatomical_label), start, end, branch_id))
        meta = ret['meta']
        ret = {
            "#reference": airway.astype(np.uint8),
            "#all_labeled_reference": relabel_all_label_airway.astype(np.int16),
            "meta": meta,
            "adj": adj.astype(np.uint8),
            "branch_info": new_branch_infos
        }
        return ret


class ChunkAirway18LabelsTest(Dataset):

    def __init__(self, input_path, keep_sorted=True):
        super(ChunkAirway18LabelsTest, self).__init__()
        self.keep_sorted = keep_sorted
        self.input_path = input_path
        all_files = glob.glob(input_path + '/*.mhd')
        self.series_uids_file_map = {Path(f).stem: f for f in all_files}
        self.series_uids = list(self.series_uids_file_map.keys())

    def __len__(self):
        return len(self.series_uids)

    def read(self, series_uid):
        f_path = self.series_uids_file_map[series_uid]
        return sitk.ReadImage(f_path)

    def build_wave_front_tree(self, all_label_mask):
        # use only the largest connection component
        struct_e = ndimage.generate_binary_structure(3, 2)
        labeled, cc_num = ndimage.label(all_label_mask > 0, struct_e)
        max_id = np.bincount(labeled.flat)[1:].argmax()
        largest_cc = labeled == (max_id + 1)
        all_label_mask = all_label_mask * largest_cc
        # get rid of small labels.
        old_labels = np.unique(all_label_mask)[1:]
        r_all_label_mask = remove_small_objects(all_label_mask, min_size=5, in_place=False)
        cut_region = np.logical_xor(r_all_label_mask > 0, all_label_mask > 0)

        label_cut_region, cc_num = ndimage.label(cut_region, struct_e)
        for label in range(1, cc_num + 1):
            cut_b = label_cut_region == label
            ol, nl = vote_region_based_on_neighbors(all_label_mask, cut_b, 2)
            print("vote cut region {} -> {}.".format(ol, nl), flush=True)

        # sanity check
        r_all_label_mask = remove_small_objects(all_label_mask, min_size=5, in_place=False)
        cut_region = np.logical_xor(r_all_label_mask > 0, all_label_mask > 0)
        assert cut_region.sum() == 0
        labels = np.unique(all_label_mask)[1:]

        print("remove small labels : {} objects -> {}.".format(len(old_labels), len(labels)),
              flush=True)
        # for each label, get rid of small cc.
        object_slices = ndimage.find_objects(all_label_mask)
        for idx, object_slice in enumerate(object_slices):
            if object_slice == None:
                continue
            label = idx + 1
            al_chunk = all_label_mask[object_slice]
            al_chunk_b = al_chunk == label
            al_chunk_labeled, cc_num = ndimage.label(al_chunk_b, struct_e)
            if cc_num != 1:
                _lb, _ls = np.unique(al_chunk_labeled, return_counts=True)
                print("bad cc occurs: {} label has {} cc {} of sizes {}.".format(label, cc_num,
                                                                                 _lb[1:], _ls[1:]), flush=True)

                for i in np.argsort(_ls[1:])[:-1]:
                    bbi = al_chunk_labeled == _lb[i + 1]
                    ol, nl = vote_region_based_on_neighbors(al_chunk, bbi, 2)
                    print("vote smaller cc {} -> {}.".format(ol, nl), flush=True)

        # relabel the label map so the labels are sorted, starting at 1.
        labels = np.unique(all_label_mask)[1:]
        relabeled_all_label = np.zeros_like(all_label_mask, np.int)
        for idx, label in enumerate(sorted(labels)):
            relabeled_all_label[all_label_mask == label] = idx + 1
        all_label_mask = relabeled_all_label
        labels = np.unique(all_label_mask)[1:]
        print("relabled {} labels here.".format(len(labels)), flush=True)
        # step.2 now we build tree
        coordinates = np.asarray(np.where(all_label_mask > 0)).T
        kd_tree = spatial.cKDTree(coordinates)
        knn_dist, knn_cds = kd_tree.query(coordinates, k=26, distance_upper_bound=2)
        knn_cds_dict = {tuple(c): tuple(coordinates[e[np.nonzero(kdst != np.inf)]].T)
                        for c, kdst, e in zip(coordinates, knn_dist, knn_cds)}
        all_label_slices = ndimage.find_objects(all_label_mask)
        assert len(list(all_label_slices)) == len(labels)

        label_center_cache = {}
        g = nx.Graph()
        v_adj = np.zeros((len(labels), len(labels)))
        for idx, all_label_slice in enumerate(all_label_slices):
            label = idx + 1
            b_mask = all_label_mask == label
            b_cds = np.asarray(np.where(b_mask > 0)).T
            b_slices = tuple([slice(max(0, ss.start - 3), min(ss.stop + 3, sp))
                              for ss, sp in zip(all_label_slice, all_label_mask.shape)])
            b_mask_chunk = b_mask[b_slices]
            _, cc_num = ndimage.label(b_mask_chunk, struct_e)
            assert cc_num == 1
            b_mask_sk = skeletonize_3d(b_mask_chunk)
            if np.sum(b_mask_sk) == 0:
                b_mask_chunk_coor = np.asarray(np.where(b_mask_chunk > 0))
                tls, brs = np.min(b_mask_chunk_coor, 1), np.max(b_mask_chunk_coor, 1)
                diameter = float(np.max([br - tl for tl, br in zip(tls, brs)]))
                b_mask_center = tuple(np.mean(b_mask_chunk_coor, 1).astype(np.int16))
                if not b_mask_chunk[b_mask_center] > 0:
                    b_mask_chunk_dist = ndimage.distance_transform_edt(b_mask_chunk)
                    b_mask_center = np.unravel_index(np.argmax(b_mask_chunk_dist, axis=None), b_mask_chunk_dist.shape)
                    assert b_mask_chunk[b_mask_center] > 0
            else:
                sub_g = make_graph_skeleton(b_mask_sk)
                e_sub_g = nx.eccentricity(sub_g)
                diameter = max(e_sub_g.values())
                b_mask_center = nx.center(sub_g)[0]
                sub_g.clear()
            # b_mask_dist = ndimage.distance_transform_edt(b_mask[b_slices])
            # b_mask_center = np.unravel_index(np.argmax(b_mask_dist, axis=None), b_mask_dist.shape)
            top_left = [ss.start for ss in b_slices]
            b_mask_center = tuple((np.asarray(b_mask_center) + top_left).tolist())
            label_center_cache[label] = b_mask_center
            assert all_label_mask[b_mask_center] == label
            # query node labels in binary mask.
            b_knn = [all_label_mask[knn_cds_dict[tuple(b_cd)]].tolist() for b_cd in b_cds]
            neighbor_labels = list(set(np.unique([xx for x in b_knn for xx in x])) - {0, label})
            # print("neighbor_labels of {}: {}".format(label, neighbor_labels), flush=True)
            assert len(neighbor_labels) > 0
            # assert len(neighbor_labels) <= 3 and len(neighbor_labels) >= 1
            g.add_node(label - 1, point=b_mask_center, length=diameter)
            for nl in neighbor_labels:
                if nl in label_center_cache.keys():
                    g.add_edge(label - 1, nl - 1,
                               distance=spatial.distance.euclidean(
                                   label_center_cache[nl],
                                   label_center_cache[label])
                               )
                else:
                    g.add_edge(label - 1, nl - 1)
                v_adj[label - 1, nl - 1] = 1
                v_adj[nl - 1, label - 1] = 1
            v_adj[label - 1, label - 1] = 1

        assert g.number_of_nodes() == len(labels)
        assert nx.is_connected(g)
        try:
            cycles = nx.find_cycle(g, orientation='ignore')
            print("found {} cycles.".format(cycles), flush=True)
            g = nx.minimum_spanning_tree(g)
        except Exception:
            print("no cycle is found!", flush=True)

        assert nx.is_tree(g)
        adj = nx.to_numpy_array(g, nodelist=list(range(len(labels))))
        adj = np.eye(adj.shape[0]) + adj
        print("ADJ check sum: {}".format((v_adj - adj).sum()), flush=True)
        # build branch record list
        branch_records = [(g.nodes[n]['point'], g.nodes[n]['length'],
                           g.nodes[n]['point'], g.nodes[n]['point'], n + 1) for n in sorted(g.nodes())]

        return adj, g, branch_records, all_label_mask

    def label_main_brochial(self, wavefront_ref_map, adj):
        reverse_wavefront_ref_map = {v: k for k, v in wavefront_ref_map.items()}
        adj_no_self = adj - np.eye(adj.shape[0])
        adj_uh = np.triu(adj_no_self)
        d_g = nx.DiGraph(adj_uh)
        RMB_label = nx.shortest_path(d_g, 0, reverse_wavefront_ref_map[2] - 1)[1] + 1
        assert (RMB_label == nx.shortest_path(d_g, 0, reverse_wavefront_ref_map[3] - 1)[1] + 1)

        LMB_label = nx.shortest_path(d_g, 0, reverse_wavefront_ref_map[13] - 1)[1] + 1
        assert (LMB_label == nx.shortest_path(d_g, 0, reverse_wavefront_ref_map[14] - 1)[1] + 1)
        return RMB_label, LMB_label

    def __getitem__(self, scan_index):
        return self.get_data(self.series_uids[scan_index])

    def get_data(self, series_uid):

        seg_airway_chunk_labeled = self.read(series_uid)
        base_dict = {
            'uid': series_uid,
        }
        base_dict.update({"size": seg_airway_chunk_labeled.GetSize()[::-1],
                          "spacing": seg_airway_chunk_labeled.GetSpacing()[::-1],
                          "original_spacing": seg_airway_chunk_labeled.GetSpacing()[::-1],
                          "original_size": seg_airway_chunk_labeled.GetSize()[::-1],
                          "origin": seg_airway_chunk_labeled.GetOrigin()[::-1],
                          "direction": np.asarray(seg_airway_chunk_labeled.GetDirection()).reshape(3, 3)[
                                       ::-1].flatten().tolist()})
        all_label_airway = sitk.GetArrayFromImage(seg_airway_chunk_labeled).astype(np.int16)
        ret = {
            "#all_labeled_reference": all_label_airway,
            "meta": base_dict,
        }

        ret = Resample('fixed_spacing', (0.625, 0.5, 0.5))(ret)
        all_label_airway = ret['#all_labeled_reference']
        adj, graph, branch_infos, relabel_all_label_airway = self.build_wave_front_tree(all_label_airway)
        new_branch_infos = []
        for x in branch_infos:
            new_branch_infos.append((x[0],x[1],x[-1],x[2],x[3],x[4]))
        meta = ret['meta']
        ret = {
            "#all_labeled_reference": relabel_all_label_airway.astype(np.int16),
            "meta": meta,
            "adj": adj.astype(np.uint8),
            "branch_info": new_branch_infos
        }
        return ret

