from cnn.neural_network import HistoNet
import torch
from algorithms.algo_utils import load_object, save_object
from os.path import join
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
from os.path import isfile
import nibabel as nib


class FullImageEstimate:

    def __init__(self, cnn, processed_brainslice):

        self.mr_img =

        # register segmentation to determine wether a patch should be presented to the CNN
        # Extract patch
        # Use patch creator ?


if __name__ == '__main__':
    output_folder = '/Users/arnaud.marcoux/histo_mri/pickled_data'

    patch_aggregator = load_object(join(output_folder, 'patch_aggregator'))

    dataset_train = patch_aggregator.get_tensor(*['TG03', 'TG05', 'TG06', 'WT03', 'WT05', 'WT06'])
    dataset_test = patch_aggregator.get_tensor(*['TG04', 'WT04'])

    if not isfile(join(output_folder, 'image_estimation', 'cnn')):
        cnn = HistoNet()

        best_params = {'batch_size': 64,
                       'shuffle': True,
                       'num_workers': 8}
        dataloader = DataLoader(dataset_train, **best_params)
        cnn.train_nn(dataloader, lr=0.00316228, n_epoch=15, val_data=dataset_test)
        save_object(cnn, join(output_folder, 'image_estimation', 'cnn'))
    else:
        cnn = load_object(join(output_folder, 'image_estimation', 'cnn'))



