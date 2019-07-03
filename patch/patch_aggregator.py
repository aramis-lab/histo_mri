from input.preprocessed_brain_slice import PreprocessedBrainSlice
from algorithms.inter_modality_matching import InterModalityMatching
from patch.patch_creator import PatchCreator
from algo_utils import save_object, load_object


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

        self.all_patches = []
        self.all_labels = []

        for patch_creator in args:
            self.all_patches.extend(patch_creator.input_patches)
            self.all_labels.extend(patch_creator.labels)

        # Check consistency
        assert len(self.all_labels) == len(self.all_patches), 'Number of labels != number of patches'


if __name__ == '__main__':

    patch_shape = (32, 32)

    tg03 = PreprocessedBrainSlice('/Users/arnaud.marcoux/histo_mri/images/TG03')
    realignment_tg03 = InterModalityMatching(tg03, create_new_transformation=False)
    patches_tg03 = PatchCreator(tg03, realignment_tg03, patch_shape)

    wt03 = PreprocessedBrainSlice('/Users/arnaud.marcoux/histo_mri/images/WT03')
    realignment_wt03 = InterModalityMatching(wt03, create_new_transformation=False)
    patches_wt03 = PatchCreator(wt03, realignment_wt03, patch_shape)

    wt04 = PreprocessedBrainSlice('/Users/arnaud.marcoux/histo_mri/images/WT04')
    realignment_wt04 = InterModalityMatching(wt04, create_new_transformation=False)
    patches_wt04 = PatchCreator(wt04, realignment_wt04, patch_shape)

    wt05 = PreprocessedBrainSlice('/Users/arnaud.marcoux/histo_mri/images/WT05')
    realignment_wt05 = InterModalityMatching(wt05, create_new_transformation=False)
    patches_wt05 = PatchCreator(wt05, realignment_wt05, patch_shape)

    wt06 = PreprocessedBrainSlice('/Users/arnaud.marcoux/histo_mri/images/WT06')
    realignment_wt06 = InterModalityMatching(wt06, create_new_transformation=False)
    patches_wt06 = PatchCreator(wt06, realignment_wt06, patch_shape)

    save_object(patches_tg03, '/Users/arnaud.marcoux/histo_mri/pickled_data/tg03/patches')
    save_object(patches_wt03, '/Users/arnaud.marcoux/histo_mri/pickled_data/wt03/patches')
    save_object(patches_wt04, '/Users/arnaud.marcoux/histo_mri/pickled_data/wt04/patches')
    save_object(patches_wt05, '/Users/arnaud.marcoux/histo_mri/pickled_data/wt05/patches')
    save_object(patches_wt06, '/Users/arnaud.marcoux/histo_mri/pickled_data/wt06/patches')

    mri_patches = PatchAggregator(patches_tg03, patches_wt03, patches_wt04, patches_wt05, patches_wt06)
    save_object(mri_patches, '/Users/arnaud.marcoux/histo_mri/pickled_data/aggregator_test')
