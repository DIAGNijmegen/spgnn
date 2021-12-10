from torch.utils.data.sampler import Sampler
import numpy as np
import random
from sklearn.model_selection import StratifiedShuffleSplit



class TensorChunkSetSampler(Sampler):

    def __init__(self, data_set, augment_size_factor, fitness_func, max_num=5, split_size=4):
        super(TensorChunkSetSampler, self).__init__(data_set)
        self.data_set = data_set
        # ref_tensor_idx indicates which tensor stores actually reference standard.
        self.ref_tensor_idx = self.data_set.ref_idx
        self.fitness_func = fitness_func
        self.augment_size_factor = augment_size_factor
        self.split_size = split_size
        self.max_num = max_num
        self.sampling_selection = self.get_indices()

    def ensure_minimal_samples(self, sampling_indices, scores):
        y = np.asarray([str(scores[x]) for x in sampling_indices])
        y_labels, y_counts = np.unique(y, return_counts=True)
        for y_label, y_count in zip(y_labels, y_counts):
            if y_count < self.split_size:
                diff = self.split_size - y_count
                label_indices = [x for x in sampling_indices if str(scores[x]) == y_label]
                sampling_indices.extend(np.random.choice(label_indices, diff).tolist())
        return sampling_indices

    def get_indices(self):

        # this function helps to sample according to the fitness value of each instance.
        # the sample selection depends on the expecting sample size according to {self.augment_size_factor}.
        # and the predefined fitness function {self.fitness_func}.
        window_locations_list = self.data_set.window_locations_list
        tensors = self.data_set.tensors_list[self.ref_tensor_idx].cpu().numpy()
        fitness_scores = []
        idx = 0
        for b_id, window_locations in enumerate(window_locations_list):
            for window_location in window_locations:
                chunk_slices = tuple([slice(center - res // 2, center + res - res // 2)
                                      for res, center in zip(self.data_set.resolutions[self.ref_tensor_idx],
                                                             window_location)])
                score = self.fitness_func(tensors[b_id][chunk_slices])
                fitness_scores.append((idx, chunk_slices, score))
                idx += 1
        indices, slices, scores = zip(*fitness_scores)
        expected_sample_size = int(len(scores) * self.augment_size_factor)
        sampling_indices = random.sample(indices, expected_sample_size)
        sampling_indices = self.ensure_minimal_samples(sampling_indices, scores)
        return [(int(x), slices[x], scores[x]) for x in sampling_indices]

    def draw_samples(self):
        # now we reshuffle sample selection so it orders in batches. the number of
        # samples in each batch is simply #sample_selection / batch_size.
        # by using StratifiedShuffleSplit, we ensure that each batch distribute similarly to the sample selection
        # in terms of the fitness scores.
        X = np.zeros((len(self.sampling_selection), 1))
        y = np.asarray([str(x[-1]) for x in self.sampling_selection])
        if len(np.unique(y)) <= 1:
            return list(range(len(self.sampling_selection)))
        test_size = int(max(len(np.unique(y)) * 2, self.split_size))
        if test_size >= len(y):
            test_size = max(2, len(y) // 2)
        n_splits = len(self.sampling_selection) // test_size
        # sample batches using stratified sampler, iteratively finding the optimal n_splits
        s = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
        indices = np.asarray([test_index for _, test_index in s.split(X, y)]).flatten().tolist()
        # print([self.sampling_selection[x][-1] for x in indices])
        return indices

    def __iter__(self):
        return iter([self.sampling_selection[i][0] for i in self.draw_samples()])

    def __len__(self):
        return len(self.sampling_selection)

class TensorChunkSetLabelFrequencyTypeSampler(TensorChunkSetSampler):

    def __init__(self, data_set, augment_size_factor, fitness_func, split_size=4):
        super(TensorChunkSetLabelFrequencyTypeSampler, self) \
            .__init__(data_set, augment_size_factor, fitness_func, split_size)

    def get_indices(self):
        # this function helps to sample according to the fitness value of each instance.
        # the sample selection depends on the expecting sample size according to {self.augment_size_factor}.
        # and the predefined fitness function {self.fitness_func}.
        window_types_list = self.data_set.window_locations_type
        types_flat = np.asarray(window_types_list).flatten()
        indices = list(range(len(types_flat)))
        unique_types = np.unique(types_flat)
        # group scores for balanced sampling, later this balanced sampling can be useful for stratefied sampling.
        indices_groups = [np.asarray(indices)[np.where(types_flat == us)] for us in unique_types]
        expected_sample_size = int(len(types_flat) * self.augment_size_factor)
        sampled_indices_groups = [np.random.choice(ig.tolist(), expected_sample_size // len(unique_types))
                                  for ig in indices_groups]
        sampling_indices = np.asarray(sampled_indices_groups).flatten().tolist()
        sampling_indices = self.ensure_minimal_samples(sampling_indices, types_flat)
        return [(x, types_flat[x]) for x in sampling_indices]

class DeepClusterSampler(Sampler):
    def __init__(self, data_source, batch_size, hyper_dict, logger, minimal_label_count=None):
        super(DeepClusterSampler, self).__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        label_mapping = hyper_dict['uids_assignment_dict']
        self.minimal_label_count = minimal_label_count
        self.logger = logger
        assert len(label_mapping.keys()) == len(data_source.series_uids)
        all_idx = list(range(len(data_source.series_uids)))
        all_scores = [label_mapping[uid] for uid in data_source.series_uids]
        self.logger.info("DeepClusterSampler: total {} instances to sample from."
                         .format(len(all_idx)))
        labels, counts = np.unique(all_scores, return_counts=True)
        if self.minimal_label_count is None:
            self.minimal_label_count = int(np.median(counts))

        selected_indices = []
        for label, count in zip(labels, counts):
            al_indices = np.where(np.asarray(all_scores) == label)[0]
            sampled_indices = np.random.choice(al_indices, self.minimal_label_count,
                                               replace=self.minimal_label_count > len(al_indices))
            selected_indices.extend(sampled_indices)

        selected_scores = [all_scores[x] for x in selected_indices]
        X = np.zeros((len(selected_scores), 1))
        y = np.asarray(selected_scores)
        test_size = min(max(self.batch_size, len(np.unique(selected_scores))), len(selected_scores) // 2)
        n_splits = len(selected_scores) // test_size
        s = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
        shuffle_indices = np.asarray([test_index for _, test_index in s.split(X, y)]).flatten().tolist()

        self.indices = [all_idx[selected_indices[i]] for i in shuffle_indices]
        sampled_scores = [label_mapping[data_source.series_uids[x]] for x in self.indices]

        k = min(20, len(sampled_scores))
        self.logger.info("DeepClusterSampler sampled distribution: {}, "
                         "sequence {} first {} items.".format(np.unique(sampled_scores, return_counts=True),
                                                              sampled_scores[:k], k))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
