from input.preprocessed_brain_slice import PreprocessedBrainSlice
from algorithms.inter_modality_matching import InterModalityMatching
from patch.patch_creator import PatchCreator
from algorithms.algo_utils import save_object, load_object
from torch.utils.data.dataset import TensorDataset
import torch
from collections import Counter
import numpy as np


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
        n_2 = label_count[2]
        description = ' * PatchAgregator * \nNumber of samples : ' + str(self.all_patches.shape[0]) \
                      + '\nLabels description - dnf : ' + str(n_1) + ' no dnf : ' + str(n_2)
        return description

    def get_tensor(self, *args):
        matching_idx = [i for i, name in enumerate(self.mouse_name) if name in args]
        matching_patches = self.all_patches[matching_idx]
        matching_labels = self.all_labels[matching_idx]
        return TensorDataset(torch.from_numpy(matching_patches),
                             torch.from_numpy(matching_labels))


if __name__ == '__main__':

    # patch_shape = (32, 32)
    #
    # tg03 = PreprocessedBrainSlice('/Users/arnaud.marcoux/histo_mri/images/TG03')
    # realignment_tg03 = InterModalityMatching(tg03, create_new_transformation=False)
    # patches_tg03 = PatchCreator(tg03, realignment_tg03, patch_shape)
    #
    # wt03 = PreprocessedBrainSlice('/Users/arnaud.marcoux/histo_mri/images/WT03')
    # realignment_wt03 = InterModalityMatching(wt03, create_new_transformation=False)
    # patches_wt03 = PatchCreator(wt03, realignment_wt03, patch_shape)
    #
    # wt04 = PreprocessedBrainSlice('/Users/arnaud.marcoux/histo_mri/images/WT04')
    # realignment_wt04 = InterModalityMatching(wt04, create_new_transformation=False)
    # patches_wt04 = PatchCreator(wt04, realignment_wt04, patch_shape)
    #
    # wt05 = PreprocessedBrainSlice('/Users/arnaud.marcoux/histo_mri/images/WT05')
    # realignment_wt05 = InterModalityMatching(wt05, create_new_transformation=False)
    # patches_wt05 = PatchCreator(wt05, realignment_wt05, patch_shape)
    #
    # wt06 = PreprocessedBrainSlice('/Users/arnaud.marcoux/histo_mri/images/WT06')
    # realignment_wt06 = InterModalityMatching(wt06, create_new_transformation=False)
    # patches_wt06 = PatchCreator(wt06, realignment_wt06, patch_shape)
    #
    # save_object(patches_tg03, '/Users/arnaud.marcoux/histo_mri/pickled_data/tg03/patches')
    # save_object(patches_wt03, '/Users/arnaud.marcoux/histo_mri/pickled_data/wt03/patches')
    # save_object(patches_wt04, '/Users/arnaud.marcoux/histo_mri/pickled_data/wt04/patches')
    # save_object(patches_wt05, '/Users/arnaud.marcoux/histo_mri/pickled_data/wt05/patches')
    # save_object(patches_wt06, '/Users/arnaud.marcoux/histo_mri/pickled_data/wt06/patches')
    #
    # mri_patches = PatchAggregator(patches_tg03, patches_wt03, patches_wt04, patches_wt05, patches_wt06)
    # save_object(mri_patches, '/Users/arnaud.marcoux/histo_mri/pickled_data/aggregator_test')

    patches_tg03 = load_object('/Users/arnaud.marcoux/histo_mri/pickled_data/tg03/patches')
    patches_wt03 = load_object('/Users/arnaud.marcoux/histo_mri/pickled_data/wt03/patches')
    patches_wt04 = load_object('/Users/arnaud.marcoux/histo_mri/pickled_data/wt04/patches')
    patches_wt05 = load_object('/Users/arnaud.marcoux/histo_mri/pickled_data/wt05/patches')
    patches_wt06 = load_object('/Users/arnaud.marcoux/histo_mri/pickled_data/wt06/patches')
    # mri_patches = load_object('/Users/arnaud.marcoux/histo_mri/pickled_data/aggregator_test')

    mri_patches = PatchAggregator(patches_tg03, patches_wt03, patches_wt04, patches_wt05, patches_wt06)

