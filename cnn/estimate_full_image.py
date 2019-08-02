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
import os
from patch.patch_aggregator import PatchAggregator


class FullImageEstimate:

    def __init__(self, cnn_histo, patch_creator, realignment, mri_shape):

        labels_estimate = np.argmax(cnn_histo(torch.tensor(patch_creator.input_patches,
                                                           dtype=torch.float32)).detach().numpy(),
                                    axis=1)

        self.image_name = patch_creator.name

        # This section obtains the estimation from the CNN HistoNet
        self.probability_map_dnf = np.zeros(mri_shape)
        self.probability_map_no_dnf = np.zeros(mri_shape)
        for lab, coord in zip(labels_estimate, patch_creator.mri_coordinates):
            # 2 - lab is used to obtain the following look up table
            # 0 background
            # 1 dnf
            # 2 no-dnf
            patch_location = Image.new('1', mri_shape, color=0)
            ImageDraw.Draw(patch_location).polygon(list(((tuple(co) for co in coord))),
                                                   outline=1,
                                                   fill=1)
            idx = np.where(np.array(patch_location))
            if lab == 0:
                self.probability_map_no_dnf[idx] += 1
            elif lab == 1:
                self.probability_map_dnf[idx] += 1

        self.probability_map_no_dnf = np.transpose(self.probability_map_no_dnf)
        self.probability_map_dnf = np.transpose(self.probability_map_dnf)

        self.probability_map_no_dnf /= np.max(self.probability_map_no_dnf)
        self.probability_map_dnf /= np.max(self.probability_map_dnf)

        self.warped_labels_ground_truth = warp(np.load(patch_creator.labelized_img),
                                               realignment.transformation_matrix,
                                               output_shape=mri_shape)

        # This section obtains a ground truth based on the real label
        self.ground_truth_from_patches = np.zeros(mri_shape)
        self.gt_dnf = np.zeros(mri_shape)
        self.gt_no_dnf = np.zeros(mri_shape)
        for lab, coord in zip(patch_creator.labels, patch_creator.mri_coordinates):
            patch_location = Image.new('1', mri_shape, color=0)
            ImageDraw.Draw(patch_location).polygon(list(((tuple(co) for co in coord))),
                                                   outline=1,
                                                   fill=1)
            idx = np.where(np.array(patch_location))
            if lab == 2:
                self.gt_no_dnf[idx] += 1
            elif lab == 1:
                self.gt_dnf[idx] += 1

        self.gt_no_dnf = np.transpose(self.gt_no_dnf)
        self.gt_dnf = np.transpose(self.gt_dnf)

        self.gt_no_dnf /= np.max(self.gt_no_dnf)
        self.gt_dnf /= np.max(self.gt_dnf)

    @staticmethod
    def produce_estimate(mri_shape, labels, mri_coordinates):
        pass

    def show_estimate(self, output_dir):

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        myfig = plt.figure(3)
        myfig.suptitle('Estimation of labels in ' + self.image_name, fontsize=7)

        f, ax1 = plt.subplot(2, 3, 1)
        plt.imshow(self.probability_map_no_dnf)
        plt.title('Probability map NO DNF', fontsize=7)
        ax1[0, 0].axis('off')

        f, ax2 = plt.subplot(2, 3, 2)
        plt.imshow(self.probability_map_dnf)
        plt.title('Probability map DNF', fontsize=7)
        ax2[0, 0].axis('off')

        f, ax3 = plt.subplot(2, 3, 3)
        plt.imshow(np.stack([self.probability_map_dnf,
                             self.probability_map_no_dnf,
                             np.zeros(self.probability_map_dnf.shape)],
                            axis=2))
        plt.title('Combined probability', fontsize=7)
        ax3[0, 0].axis('off')

        f, ax4 = plt.subplot(2, 3, 4)
        plt.imshow(self.warped_labels_ground_truth)
        plt.title('labelized image warped to MR', fontsize=7)
        ax4[0, 0].axis('off')

        f, ax5 = plt.subplot(2, 3, 5)
        plt.imshow(np.stack([self.gt_dnf,
                             self.gt_no_dnf,
                             np.zeros((384, 384))],
                            axis=2))
        plt.title('Ground truth')
        ax5[0, 0].axis('off')

        plt.savefig(join(output_dir, self.image_name + '_estimation.png'), dpi=600)


if __name__ == '__main__':
    output_folder = '/Users/arnaud.marcoux/histo_mri/pickled_data'

    patch_aggregator = load_object(join(output_folder, 'patch_aggregator_8_8'))

    dataset_train = patch_aggregator.get_tensor(*['TG03', 'TG05', 'WT03', 'WT05'])
    dataset_test = patch_aggregator.get_tensor(*['TG04', 'WT06'])

    if not isfile(join(output_folder, 'image_estimation', 'cnn')):
        cnn = HistoNet()

        best_params = {'batch_size': 64,
                       'shuffle': True,
                       'num_workers': 8}
        dataloader = DataLoader(dataset_train, **best_params)
        cnn.train_nn(dataloader, lr=0.0025118864315095794, n_epoch=12, val_data=dataset_test)
        save_object(cnn, join(output_folder, 'image_estimation', 'cnn'))
    else:
        cnn = load_object(join(output_folder, 'image_estimation', 'cnn'))

    realignements = load_object(join(output_folder, 'realignements'))
    patch_creators = load_object(join(output_folder, 'patch_creators_8_8'))

    # Choose what slice to display
    # n = 0 -> TG03
    # n = 1 -> TG04
    # n = 2 -> TG05
    # n = 3 -> TG06
    # n = 4 -> WT03
    # n = 5 -> WT04
    # n = 6 -> WT05
    # n = 7 -> WT06

    for n in [1, 7]:
        img_estimate = FullImageEstimate(cnn, patch_creators[n], realignements[n], (384, 384))
        img_estimate.show_estimate(output_folder)
