from cnn.neural_network import HistoNet
from algorithms.algo_utils import load_object, save_object
from os.path import join
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch
from torch.utils.data.dataloader import DataLoader
from os.path import isfile, dirname
import numpy as np
from skimage.transform import warp
import os
from patch.patch_aggregator import PatchAggregator


class FullImageEstimate:

    def __init__(self, cnn_histo, patch_creator, realignment, mri_shape):

        labels_estimate = cnn_histo.predict(patch_creator.input_patches)
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

        self.image_estimation = self.pixel_wise_classification(patch_creator.mri_coordinates,
                                                               labels_estimate,
                                                               mri_shape)
        self.ground_truth_pixelwise = self.pixel_wise_classification(patch_creator.mri_coordinates,
                                                                     patch_creator.labels,
                                                                     mri_shape)

    @staticmethod
    def produce_estimate(mri_shape, labels, mri_coordinates):
        pass

    def show_estimate(self, output_file):

        if not os.path.exists(dirname(output_file)):
            os.mkdir(dirname(output_file))

        myfig = plt.figure(3)
        myfig.suptitle('Estimation of labels in ' + self.image_name, fontsize=7)

        ax = plt.subplot(3, 2, 1)
        plt.imshow(self.probability_map_no_dnf)
        plt.title('Probability map NO DNF', fontsize=7)
        ax.axis('off')

        ax = plt.subplot(3, 2, 2)
        plt.imshow(self.probability_map_dnf)
        plt.title('Probability map DNF', fontsize=7)
        ax.axis('off')

        # ax = plt.subplot(4, 2, 4)
        # plt.imshow(self.warped_labels_ground_truth)
        # plt.title('labelized image warped to MR', fontsize=7)
        # ax.axis('off')

        ax = plt.subplot(3, 2, 3)
        plt.imshow(np.stack([self.probability_map_dnf,
                             self.probability_map_no_dnf,
                             np.zeros(self.probability_map_dnf.shape)],
                            axis=2))
        plt.title('2 maps of probability stacked', fontsize=7)
        ax.axis('off')

        ax = plt.subplot(3, 2, 4)
        plt.imshow(np.stack([self.gt_dnf,
                             self.gt_no_dnf,
                             np.zeros((384, 384))],
                            axis=2))
        plt.title('Ground truth probability maps', fontsize=7)
        ax.axis('off')

        ax = plt.subplot(3, 2, 5)
        plt.imshow(self.image_estimation, vmin=0, vmax=2)
        plt.title('image estimation pixelwise', fontsize=7)
        ax.axis('off')

        ax = plt.subplot(3, 2, 6)
        plt.imshow(self.ground_truth_pixelwise, vmin=0, vmax=2)
        plt.title('ground truth pixelwise', fontsize=7)
        ax.axis('off')

        plt.savefig(output_file, dpi=800)

    @staticmethod
    def pixel_wise_classification(mri_coordinates, labels_estimate, mri_shape):
        count_dnf = np.zeros(mri_shape)
        count_no_dnf = np.zeros(mri_shape)
        for lab, coord in zip(labels_estimate, mri_coordinates):
            patch_location = Image.new('1', mri_shape, color=0)
            ImageDraw.Draw(patch_location).polygon(list(((tuple(co) for co in coord))),
                                                   outline=1,
                                                   fill=1)
            idx = np.where(np.array(patch_location))
            if lab == 0 or lab == 2:
                count_no_dnf[idx] += 1
            elif lab == 1:
                count_dnf[idx] += 1
        count_total = np.stack([np.zeros(mri_shape),
                                count_no_dnf,
                                count_dnf], axis=2)
        return np.transpose(np.argmax(count_total, axis=2))


if __name__ == '__main__':
    output_folder = '/Users/arnaud.marcoux/histo_mri/pickled_data/'

    patch_aggregator = load_object(join(output_folder, 'patch_aggregator_8_8'))
    realignements = load_object(join(output_folder, 'realignements'))
    patch_creators = load_object(join(output_folder, 'patch_creators_8_8'))

    train_set = [['TG05', 'TG06', 'WT04', 'WT05'],
                 ['TG03', 'TG05', 'WT03', 'WT04'],
                 ['TG03', 'TG06', 'WT03', 'WT05']]
    val_set = [['TG03', 'WT03'],
               ['TG06', 'WT05'],
               ['TG05', 'WT04']]
    test_set = ['TG04', 'WT06']

    folds = len(train_set)

    dataset_test = patch_aggregator.get_tensor(*test_set)

    for i, training in enumerate(train_set):
        dataset_train = patch_aggregator.get_tensor(*training)
        cnn = HistoNet()
        best_params = {'batch_size': 64,
                       'shuffle': True,
                       'num_workers': 8}
        dataloader = DataLoader(dataset_train, **best_params)
        cnn.train_nn(dataloader, lr=0.1, n_epoch=2, val_data=dataset_test)
        for n in [1, 7]:
            img_estimate = FullImageEstimate(cnn, patch_creators[n], realignements[n], (384, 384))
            img_estimate.show_estimate(output_folder)

    # Choose what slice to display
    # n = 0 -> TG03
    # n = 1 -> TG04
    # n = 2 -> TG05
    # n = 3 -> TG06
    # n = 4 -> WT03
    # n = 5 -> WT04
    # n = 6 -> WT05
    # n = 7 -> WT06