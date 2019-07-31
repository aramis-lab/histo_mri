from input.preprocessed_brain_slice import PreprocessedBrainSlice
from algorithms.inter_modality_matching import InterModalityMatching
from patch.patch_creator import PatchCreator
from algorithms.algo_utils import save_object, load_object
from torch.utils.data.dataset import TensorDataset
import torch
from collections import Counter
import numpy as np
from colorama import Fore

class PatchAggregator:
    """
    PatchAggregator simplify the patch creation by providing 2 args :
    self.input_patches : all the patches aggregated
    self.labels : all the labels aggregated
    """

    # With *args, there can be an infinite number of args
    def __init__(self, *args):
        # Check that given PatchCreators all have the labels generated
        for p in args:
            assert hasattr(p, 'labels'), 'labels attribute could not be found'

        self.all_patches = np.concatenate([patch.input_patches for patch in args])

        # Unravel a list of list into a single list
        # [inner
        #     ... for outer in x
        #                ... for inner in outer]
        self.all_labels = [j for i in [patch.labels for patch in args] for j in i]
        self.all_labels = np.array(self.all_labels)
        self.all_labels[self.all_labels == 2] = 0

        # Check consistency
        if len(self.all_labels) != len(self.all_patches):
            raise ValueError('Number of labels (' + str(len(self.all_labels))
                             + ') != number of patches(' + str(len(self.all_patches)) + ')')

        self.mouse_name = []
        for m in args:
            self.mouse_name.extend([m.name] * len(m.labels))

    def __repr__(self):
        label_count = Counter(self.all_labels)
        n_1 = label_count[1]
        n_2 = label_count[0]
        description = Fore.RED + ' ** PatchAgregator ** ' + Fore.RESET + '\n\tNumber of samples : ' \
                      + str(self.all_patches.shape[0]) + '\n\tLabels description - dnf : ' + str(n_1) \
                      + ' no dnf : ' + str(n_2) + '\n\n'
        for current_name in sorted(list(set(self.mouse_name))):
            current_labels = [lab for slice_name, lab in zip(self.mouse_name, self.all_labels)
                              if slice_name == current_name]
            description += '\n' + Fore.GREEN + current_name + Fore.RESET
            description += '\ndnf count : ' + str(np.sum(current_labels))
            description += '\nno - dnf count : ' + str(len(current_labels) - np.sum(current_labels))

        return description

    def get_tensor(self, *args):
        matching_idx = [i for i, name in enumerate(self.mouse_name) if name in args]
        matching_patches = self.all_patches[matching_idx]
        matching_labels = self.all_labels[matching_idx]
        return TensorDataset(torch.from_numpy(matching_patches),
                             torch.from_numpy(matching_labels))


if __name__ == '__main__':
    patches = load_object('/Users/arnaud.marcoux/histo_mri/pickled_data/patch_creators_8_8')
    aggregator = PatchAggregator(*patches)
