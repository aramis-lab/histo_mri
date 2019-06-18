from input.preprocessed_brain_slice import PreprocessedBrainSlice
from algorithms.inter_modality_matching import InterModalityMatching

class PatchCreator:

    def __init__(self, processed_brainslice, mod_matching, patch_shape):
        pass


if __name__ == '__main__':
    tg03 = PreprocessedBrainSlice('/Users/arnaud.marcoux/histo_mri/images/TG03')
    realignment = InterModalityMatching(tg03, create_new_transformation=False)
    patches = PatchCreator(tg03, realignment)