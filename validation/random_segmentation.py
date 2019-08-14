from patch.patch_creator import PatchCreator
from input.preprocessed_brain_slice import PreprocessedBrainSlice
from algorithms.inter_modality_matching import InterModalityMatching
from scipy.ndimage.measurements import label
from multiprocessing import Pool, cpu_count
from functools import partial
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from os.path import join, basename, isdir, splitext
from os import mkdir
import uuid
import numpy as np
import random


class RandomSegmentation:

    def __init__(self, path_to_labels: str, n_segmentation: int, output_directory: str):

        generate_random_segmentation_partial = partial(self.generate_random_segmentation,
                                                       output_directory=output_directory)

        if not isdir(output_directory):
            mkdir(output_directory)

        original_segmentation_list = [path_to_labels] * n_segmentation
        pool = Pool(processes=cpu_count())
        pool.map(generate_random_segmentation_partial, original_segmentation_list)

    def perform_random_segmentation(self, original_labels: str):
        in_label = np.load(original_labels)
        labeled_mask, num_features = label(in_label == 1)
        mask_slice = in_label != 0

        new_random_segmentation = Image.fromarray(mask_slice.astype('int8'))

        unique_labels = np.unique(labeled_mask)
        pixel_count = [np.sum(labeled_mask == lab) for lab in unique_labels]
        unique_labels_sorted = [x for _, x in sorted(zip(pixel_count, unique_labels), reverse=True)]
        for count, lab in enumerate(unique_labels_sorted):
            if count != 0:
                n_px = np.sum(labeled_mask == lab)
                R = np.round(np.sqrt(n_px / np.pi))
                possible_idx = np.where(np.array(new_random_segmentation) == 1)
                try_number = 0
                while True:
                    # Must check that the circle does not overlap with existing segmentation
                    # 1 create new image, draw the circle, convert 2 numpy array
                    # 2 create mask of new_random_segmentation == 0 or 2
                    # np.sum(mask1 AND mask 2) must be 0
                    try_number += 1

                    random_draw_center = np.random.choice(np.arange(possible_idx[0].size), size=1, replace=False)
                    center_idx = tuple((possible_idx[0][random_draw_center[0]],
                                        possible_idx[1][random_draw_center[0]]))
                    choices = ['circle', 'rectangle']
                    choice = np.random.choice(choices, size=1)[0]

                    if choice == 'circle':
                        print('Drawing ' + choice + ' n# ' + str(count + 1) + '/' + str(num_features)
                              + ' of radius ' + str(int(R)) + ' at position ' + str(center_idx) + ' - try number '
                              + str(try_number))
                        current_circle = Image.new('1', (2 * int(R), 2 * int(R)), 0)
                        ImageDraw.Draw(current_circle).ellipse((0,
                                                                0,
                                                                2 * R,
                                                                2 * R),
                                                               outline=0,
                                                               fill=1)
                        current_shape_numpy = np.array(current_circle)

                        new_random_segmentation_crop = new_random_segmentation.crop((center_idx[1] - R,
                                                                                     center_idx[0] - R,
                                                                                     center_idx[1] + R,
                                                                                     center_idx[0] + R))
                        one_percent_threshold = np.round(np.pi * (R ** 2) / 100)

                    elif choice == 'rectangle':
                        # Even dimension for the bounding box are necessary to avoid dimension mismatch
                        is_even = False
                        while is_even is False:
                            ratio_H_W = random.uniform(0, 1)
                            width = np.sqrt(n_px / ratio_H_W)
                            height = np.int(ratio_H_W * width)
                            width = np.int(width)
                            theta = random.uniform(0, np.pi / 2)
                            bounding_box_rectangle = self.get_bounding_box_rectangle(width, height, theta)
                            if not any(dim % 2 != 0 for dim in bounding_box_rectangle):
                                is_even = True

                        print('Drawing ' + choice + ' n# ' + str(count + 1) + '/' + str(num_features)
                              + ' of dims ' + str((width, height)) + ' with angle ' + str(180.0 * theta / np.pi)
                              + '° at position ' + str(center_idx) + ' - try number ' + str(try_number))

                        current_rectangle = Image.new('1', bounding_box_rectangle, 0)
                        ImageDraw.Draw(current_rectangle).polygon([(0, width * np.sin(theta)),
                                                                   (height * np.sin(theta), bounding_box_rectangle[1]),
                                                                   (bounding_box_rectangle[0], bounding_box_rectangle[1] - width * np.sin(theta)),
                                                                   (width * np.cos(theta), 0)],
                                                                  outline=1,
                                                                  fill=1)
                        current_shape_numpy = np.array(current_rectangle)
                        new_random_segmentation_crop = new_random_segmentation.crop((center_idx[1] - bounding_box_rectangle[0] / 2,
                                                                                     center_idx[0] - bounding_box_rectangle[1] / 2,
                                                                                     center_idx[1] + bounding_box_rectangle[0] / 2,
                                                                                     center_idx[0] + bounding_box_rectangle[1] / 2))

                        one_percent_threshold = np.round(height * width / 100)

                    new_random_segmentation_crop_numpy = np.array(new_random_segmentation_crop)
                    new_random_segmentation_numpy_mask = new_random_segmentation_crop_numpy != 1

                    if np.sum(np.logical_and(current_shape_numpy, new_random_segmentation_numpy_mask)) <= one_percent_threshold:
                        if choice == 'circle':
                            ImageDraw.Draw(new_random_segmentation).ellipse((center_idx[1] - R,
                                                                             center_idx[0] - R,
                                                                             center_idx[1] + R,
                                                                             center_idx[0] + R),
                                                                            outline=2,
                                                                            fill=2)
                        if choice == 'rectangle':
                            left = center_idx[1] - bounding_box_rectangle[0] / 2
                            upper = center_idx[0] - bounding_box_rectangle[1] / 2
                            ImageDraw.Draw(new_random_segmentation).polygon([(left, upper + width * np.sin(theta)),
                                                                             (left + height * np.sin(theta), upper + bounding_box_rectangle[1]),
                                                                             (left + bounding_box_rectangle[0], upper + bounding_box_rectangle[1] - width * np.sin(theta)),
                                                                             (left + width * np.cos(theta), upper)],
                                                                            outline=2,
                                                                            fill=2)
                        break
                    else:
                        print('Coliding pixels must be <= ' + str(one_percent_threshold) + ' but found : '
                              + str(np.sum(np.logical_and(current_shape_numpy, new_random_segmentation_numpy_mask))))
        return np.array(new_random_segmentation, dtype='int8')

    def generate_random_segmentation(self, path_to_segmentation, output_directory):
        random_seg = self.perform_random_segmentation(path_to_segmentation)
        np.save(join(output_directory, splitext(basename(path_to_segmentation))[0] + '_' + str(uuid.uuid4())[:4], random_seg))

    @staticmethod
    def get_patch_creator(processed_brainslice: PreprocessedBrainSlice,
                          mod_matching: InterModalityMatching,
                          patch_shape: tuple,
                          labelized_img: str):
        return PatchCreator(processed_brainslice, mod_matching, patch_shape, labelized_img)

    @staticmethod
    def get_bounding_box_rectangle(width, height, theta):
        half_diag = np.sqrt(width ** 2 + height ** 2) / 2
        return (np.round(2 * half_diag * np.cos(np.arctan(height / width) - theta)).astype('int'),
                np.round(2 * half_diag * np.sin(np.arctan(height / width) + theta)).astype('int'))


if __name__ == '__main__':
    labelZ = '/Users/arnaud.marcoux/histo_mri/images/TG03/label_tg03.npy'
    output_dir = '/Users/arnaud.marcoux/histo_mri/pickled_data/random_segmentation/tg03'
    test = RandomSegmentation(labelZ, 15, output_dir)