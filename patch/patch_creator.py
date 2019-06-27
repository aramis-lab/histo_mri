from input.preprocessed_brain_slice import PreprocessedBrainSlice
from algorithms.inter_modality_matching import InterModalityMatching
from sklearn.feature_extraction import image
import numpy as np
import nibabel as nib
from colorama import Fore


class PatchCreator:
    """
    List of attributes :

    self.inputs_patches           : patches of shape (n_sample, n_channel, height, width)
    self.MRI_coordinates_patches  : list of rectangle of patches in MR coordinates
    self.histo_coordinates        : list of rectangle of patches in histo coordinates
    self.transformation_matrix    : transf. matrix so that [mr_coord] x transformation_matrix = [histo_coord]
    self.labels                   : length = n_samples, labels of corresponding rectangles
    """

    def __init__(self, processed_brainslice, mod_matching, patch_shape):
        print(Fore.GREEN + ' * Starting creation of patches for ' + processed_brainslice.name + Fore.RESET)
        self.input_patches = self.extract_patches(processed_brainslice, patch_shape)
        self.mri_coordinates = self.get_mri_patches_coordinates(processed_brainslice.mr_shape, patch_shape)

        ## Step to remove OOB patches
        # input_patches is an array, mri_coordinates is a list
        non_usable_patch = [i for i in range(self.input_patches.shape[0])
                            if np.sum(self.input_patches[i] != self.input_patches[i]) > 0]

        print(Fore.YELLOW + str(len(non_usable_patch))
              + ' patches contained np.nan object. They are not relevant for the analysis' + Fore.RESET)
        self.input_patches = np.delete(self.input_patches, non_usable_patch, axis=0)
        self.mri_coordinates = [elem for i, elem in enumerate(self.mri_coordinates) if i not in non_usable_patch]

        self.histo_coordinates = self.get_histo_patches_coordinates(self.mri_coordinates,
                                                                    mod_matching.transformation_matrix)

        oob_histo_patches = self.get_idx_oob_histo_patches(self.histo_coordinates, processed_brainslice.histo_shape)
        print(Fore.YELLOW + str(len(oob_histo_patches))
              + ' patches were out of bound when transformed into the histological space.' + Fore.RESET)
        self.input_patches = np.delete(self.input_patches, oob_histo_patches, axis=0)
        self.mri_coordinates = [elem for i, elem in enumerate(self.mri_coordinates) if i not in oob_histo_patches]

        print('Number of patches kept : ' + str(len(self.input_patches)))


    @staticmethod
    def get_idx_oob_histo_patches(histo_coordinates, img_shape):
        def is_oob_coordinate(coordinates):
            min_dim = np.min(np.array(coordinates), axis=0)
            max_dim = np.max(np.array(coordinates), axis=0)
            if any(dim < 0 for dim in min_dim) or img_shape[0] < max_dim[0] - 1 or img_shape[1] < max_dim[1] - 1:
                return False
            else:
                return True
        return [i for i, coord in enumerate(histo_coordinates) if is_oob_coordinate(coord)]

    @staticmethod
    def extract_patches(processed_brainslice, p_shape):
        """
        :param processed_brainslice: processed brainslice we want to extract patches from
        :param p_shape: shape of the desired patches
        :return: array of patches where shape is (n_samples, n_channel, patch_width, patch_height)
        """
        # Shape of img after concatenate is (number of MRI sequence, X dim, Y dim)
        # sorted() keyword is very import, as it ensures to always have the same order of sequences across the different
        # patches
        # 1. sos_mge
        # 2. sos_msme
        # 3. t1
        # 4. t2_msme
        # 5. t2s
        stacked_img = np.concatenate(tuple(np.expand_dims(nib.load(processed_brainslice.file_paths[item]).get_data(),
                                                          axis=0)
                                           for item in sorted(processed_brainslice.file_paths)),
                                     axis=0)

        # New shape of img : (x dim, y dim, number of seq)
        # This step is necessary for extract_patches from sklean
        img = np.moveaxis(stacked_img, 0, -1)

        # Shape here will be : (353, 353, 1, 32, 32, 5)
        patches = image.extract_patches(img, p_shape + (img.shape[-1],))
        # Sanity check - patches[155, 264, 0, :, :, 3]

        # When you add -1 to an axis in numpy it will just put everything else in that axis.
        # Shape is now (124609, 32, 32, 5)
        patches = patches.reshape(-1, p_shape[0], p_shape[1], img.shape[-1])
        # Sanity check - patches[155 * 353 + 264, :, :, 3]

        # Reordering in shape (124609, 5, 32, 32), but reshape here loose consistency (sanity check)
        patches = np.swapaxes(patches, 2, 3)
        patches = np.swapaxes(patches, 1, 2)
        # Sanity check - patches[155 * 353 + 264, 3, :, :]
        return patches

    @staticmethod
    def get_histo_patches_coordinates(mri_coordinates, transformation_matrix):
        """
        :param mri_coordinates: list (list of 4 coordinates) defining the rectangle of the patch
        :param transformation_matrix: transformation matrix from mri_coordinates to histo_coordinates
        :return: histo_coordinates
        """
        def transform_to_histo(coordinate):
            result = []
            for co in coordinate:
                co.append(1.0)
                result.append(tuple(np.dot(transformation_matrix,
                                           np.transpose(np.array(co)))[0:2]))
            return result

        return [transform_to_histo(coord) for coord in mri_coordinates]

    @staticmethod
    def get_mri_patches_coordinates(img_shape, patch_shape):
        """ Returns the coordinates of patches extracted with extract_patches
        method

        :param img_shape: Must be a tuple (image_height, image_width)
        :param patch_shape: Must be a tuple (patch_height, patch_width)
        :return: coordinates of the 4 points of the patch
        """
        img_height = img_shape[0]
        img_width = img_shape[1]
        patch_height = patch_shape[0]
        patch_width = patch_shape[1]

        (i, j) = (0, 0)
        coordinates = []
        while i <= img_height - patch_height:
            while j <= img_width - patch_width:
                coordinates.append([[i, j],
                                    [i, j + patch_width - 1],
                                    [i + patch_height - 1, j + patch_width - 1],
                                    [i + patch_height - 1, j]])
                j = j + 1
            j = 0
            i = i + 1
        return coordinates


if __name__ == '__main__':
    tg03 = PreprocessedBrainSlice('/Users/arnaud.marcoux/histo_mri/images/TG03')
    realignment = InterModalityMatching(tg03, create_new_transformation=False)
    pt = PatchCreator(tg03, realignment, (32, 32))
