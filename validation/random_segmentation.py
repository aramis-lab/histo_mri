from patch.patch_creator import PatchCreator
from input.preprocessed_brain_slice import PreprocessedBrainSlice
from algorithms.inter_modality_matching import InterModalityMatching
from scipy.ndimage.measurements import label
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
import random
import copy


class RandomSegmentation:

    def __init__(self, original_labels: str):
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
                    # Improvement : random draw betwen circle, square... (with different probability)
                    try_number += 1

                    random_draw_center = np.random.choice(np.arange(possible_idx[0].size), size=1, replace=False)
                    center_idx = tuple((possible_idx[0][random_draw_center[0]],
                                        possible_idx[1][random_draw_center[0]]))
                    choices = ['circle', 'rectangle']
                    choice = np.random.choice(choices, size=1)[0]
                    print('Drawing ' + choice + ' n# ' + str(count + 1) + '/' + str(num_features)
                          + ' of radius ' + str(int(R)) + ' at position ' + str(center_idx) + ' - try number '
                          + str(try_number))

                    if choice == 'circle':
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
                        new_random_segmentation_crop_numpy = np.array(new_random_segmentation_crop)
                        new_random_segmentation_numpy_mask = new_random_segmentation_crop_numpy != 1

                        one_percent_threshold = np.round(np.pi * (R ** 2) / 100)

                    if choice == 'rectangle':
                        ratio_H_W = random.uniform(0, 1)
                        width = np.sqrt(n_px / ratio_H_W)
                        height = ratio_H_W * width
                        random_rotation = random.uniform(0, 2 * np.pi)
                        
                        current_rectangle = Image.new('1',  ,0)

                    if np.sum(np.logical_and(current_shape_numpy, new_random_segmentation_numpy_mask)) <= one_percent_threshold:
                        if choice == 'circle':
                            ImageDraw.Draw(new_random_segmentation).ellipse((center_idx[1] - R,
                                                                             center_idx[0] - R,
                                                                             center_idx[1] + R,
                                                                             center_idx[0] + R),
                                                                            outline=0,
                                                                            fill=2)
                        if choice == 'rectangle':
                            ImageDraw.Draw(new_random_segmentation).ellipse((center_idx[1] - R,
                                                                             center_idx[0] - R,
                                                                             center_idx[1] + R,
                                                                             center_idx[0] + R),
                                                                            outline=0,
                                                                            fill=2)
                        break
                    else:
                        print('Coliding pixels must be <= ' + str(one_percent_threshold) + ' but found : '
                              + str(np.sum(np.logical_and(current_circle_numpy, new_random_segmentation_numpy_mask))))
        plt.imshow(np.array(new_random_segmentation))
        plt.show()

    @staticmethod
    def get_patch_creator(processed_brainslice: PreprocessedBrainSlice,
                          mod_matching: InterModalityMatching,
                          patch_shape: tuple,
                          labelized_img: str):
        return PatchCreator(processed_brainslice, mod_matching, patch_shape, labelized_img)


if __name__ == '__main__':
    labelZ = '/Users/arnaud.marcoux/histo_mri/images/TG03/label_tg03.npy'
    test = RandomSegmentation(labelZ)