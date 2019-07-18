from input.preprocessed_brain_slice import PreprocessedBrainSlice
from algorithms.inter_modality_matching import InterModalityMatching
from patch.patch_creator import PatchCreator
from patch.patch_aggregator import PatchAggregator
from cnn.neural_network import HistoNet
from algorithms.algo_utils import save_object, load_object
from os.path import join, isfile
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader


if __name__ == '__main__':

    # Var
    input_folder = '/Users/arnaud.marcoux/histo_mri/images'
    output_folder = '/Users/arnaud.marcoux/histo_mri/pickled_data'
    patch_shape = (16, 16)

    mouse_names = ['TG0' + str(i) for i in [3, 4, 5, 6]] + ['WT0' + str(i) for i in [3, 4, 5, 6]]

    if not isfile(join(output_folder, 'brain_slices')):
        brain_slices = [PreprocessedBrainSlice(join(input_folder, mouse_name)) for mouse_name in mouse_names]
        save_object(brain_slices, join(output_folder, 'brain_slices'))
    else:
        brain_slices = load_object(join(output_folder, 'brain_slices'))

    if not isfile(join(output_folder, 'realignements')):
        realignements = [InterModalityMatching(brain_slice, create_new_transformation=False) for brain_slice in brain_slices]
        save_object(realignements, join(output_folder, 'realignements'))
    else:
        realignements = load_object(join(output_folder, 'realignements'))

    if not isfile(join(output_folder, 'patch_creators')):
        patch_creators = [PatchCreator(brain_slices[i], realignements[i], patch_shape) for i in range(len(brain_slices))]
        save_object(patch_creators, join(output_folder, 'patch_creators'))
    else:
        patch_creators = load_object(join(output_folder, 'patch_creators'))

    if not isfile(join(output_folder, 'patch_aggregator')):
        patch_aggregator = PatchAggregator(*patch_creators)
        save_object(patch_aggregator, join(output_folder, 'patch_aggregator'))
    else:
        patch_aggregator = load_object(join(output_folder, 'patch_aggregator'))

    labs = patch_aggregator.all_labels
    labs[labs == 2] = 0

    histo_net_cnn = HistoNet()
    # Parameters
    params = {'batch_size': 32,
              'shuffle': True,
              'num_workers': 8}

    dataset = TensorDataset(torch.from_numpy(patch_aggregator.all_patches),
                            torch.from_numpy(labs))
    dataloader = DataLoader(dataset, **params)

    # dtype Long is necessary for labels
    histo_net_cnn.train_nn(dataloader,
                           lr=0.01,
                           n_epoch=10)
