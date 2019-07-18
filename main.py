from input.preprocessed_brain_slice import PreprocessedBrainSlice
from algorithms.inter_modality_matching import InterModalityMatching
from patch.patch_creator import PatchCreator
from patch.patch_aggregator import PatchAggregator

from os.path import join

from algorithms.algo_utils import save_object

if __name__ == '__main__':
    # Var
    input_folder = '/Users/arnaud.marcoux/histo_mri/images'
    output_folder = '/Users/arnaud.marcoux/histo_mri/pickled_data'
    patch_shape = (16, 16)

    mouse_names = ['TG0' + str(i) for i in [3, 4, 5, 6]] + ['WT0' + str(i) for i in [3, 4, 5, 6]]

    brain_slices = [PreprocessedBrainSlice(join(input_folder, mouse_name)) for mouse_name in mouse_names]
    save_object(brain_slices, join(output_folder, 'brain_slices'))

    realignements = [InterModalityMatching(brain_slice, create_new_transformation=False) for brain_slice in brain_slices]
    save_object(realignements, join(output_folder, 'realignements'))

    patch_creators = [PatchCreator(brain_slices[i], realignements[i], patch_shape) for i in range(len(brain_slices))]
    save_object(patch_creators, join(output_folder, 'patch_creators'))

    patch_aggregator = PatchAggregator(*patch_creators)
    save_object(patch_aggregator, join(output_folder, 'patch_aggregator'))
