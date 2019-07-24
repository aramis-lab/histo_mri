from cnn.neural_network import HistoNet
from algorithms.algo_utils import load_object, save_object
from os.path import join
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch
from torch.utils.data.dataloader import DataLoader
from os.path import isfile
import numpy as np
from skimage.transform import warp


class FullImageEstimate:

    def __init__(self, cnn_histo, patch_creator, realignment, mri_shape):

        labels_estimate = np.argmax(cnn_histo(torch.tensor(patch_creator.input_patches,
                                                           dtype=torch.float32)).detach().numpy(),
                                    axis=1)

        image_estimate = Image.new('L', mri_shape, color=0)
        for lab, coord in zip(labels_estimate, patch_creator.mri_coordinates):
            # lab + 1 is used to obtain the following look up table
            # 0 background
            # 1 dnf
            # 2 no-dnf
            ImageDraw.Draw(image_estimate).polygon(list(((tuple(co) for co in coord))),
                                                   outline=int(2 - lab),
                                                   fill=int(2 - lab))
        self.final_image = np.transpose(np.array(image_estimate))

        self.warped_labels_ground_truth = warp(np.load(patch_creator.labelized_img),
                                               realignment.transformation_matrix,
                                               output_shape=mri_shape)

    def show_estimate(self):
        plt.subplot(1, 2, 1)
        plt.imshow(self.final_image)
        plt.subplot(1, 2, 2)
        plt.imshow(self.warped_labels_ground_truth)
        plt.show()


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

    realignements = load_object(join(output_folder, 'realignements'))
    patch_creators = load_object(join(output_folder, 'patch_creators'))

    # Choose what slice to display
    # n = 0 -> TG03
    # n = 1 -> TG04
    # n = 2 -> TG05
    # n = 3 -> TG06
    # n = 4 -> WT03
    # n = 5 -> WT04
    # n = 6 -> WT05
    # n = 7 -> WT06

    for n in range(7):
        tg03_img_estimate = FullImageEstimate(cnn, patch_creators[n], realignements[n], (384, 384))
        tg03_img_estimate.show_estimate()