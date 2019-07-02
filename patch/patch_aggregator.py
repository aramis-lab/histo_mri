from input.preprocessed_brain_slice import PreprocessedBrainSlice
from algorithms.inter_modality_matching import InterModalityMatching
from patch.patch_creator import PatchCreator


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
            assert hasattr(p, 'labels')

        self.all_patches = []
        self.all_labels = []

        for patch_creator in args:
            self.all_patches.extend(patch_creator.input_patches)
            self.all_labels.extend(patch_creator.labels)

        # Check consistency
        assert len(self.all_labels) == len(self.all_patches)


if __name__ == '__main__':
    tg03 = PreprocessedBrainSlice('/Users/arnaud.marcoux/histo_mri/images/TG03')
    realignment_tg03 = InterModalityMatching(tg03, create_new_transformation=False)
    patches_tg03 = PatchCreator(tg03, realignment_tg03, (32, 32),
                                labelized_img='/Users/arnaud.marcoux/histo_mri/images/TG03/segmentation.npy')

    wt03 = PreprocessedBrainSlice('/Users/arnaud.marcoux/histo_mri/images/WT03')
    realignment_wt03 = InterModalityMatching(wt03, create_new_transformation=False)
    patches_wt03 = PatchCreator(wt03, realignment_wt03, (32, 32),
                                labelized_img=...)

    mri_patches = PatchAggregator(patches_tg03, patches_wt03)
