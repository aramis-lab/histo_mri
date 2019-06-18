import os
from raw_brain_slice import BrainSlice
from utils import identify_modality, coregister, normalize_vol
from algorithms.algo_utils import get_mask_data
import nibabel as nib
import numpy as np
from colorama import Fore
import warnings

warnings.filterwarnings('ignore')


class PreprocessedBrainSlice(BrainSlice):
    """
    List of attributes :
    name :          name of the brain slice
    file_paths :    dictionary representing the different modalities
    is_coregistered : bool if the brain slice have been coregistered between one MR slice to another
    coreg_reference : reference for the coregistered

    """

    def __init__(self, path):
        super(PreprocessedBrainSlice, self).__init__(path)
        self.is_coregistered, self.coreg_reference = self.check_coregistration(path)

        # Coregister and normallize data if not already done
        if not self.is_coregistered:
            if not os.path.exists(os.path.join(self.path_to_data, 'normalized')):
                self.normalize_all(threshold=500)
                self.file_paths = identify_modality(os.path.join(self.path_to_data, 'normalized'))
            self.coreg_reference = 't2s'
            self.coregister_all(self.coreg_reference)
            self.is_coregistered = True

        # Grab filenames
        self.file_paths = identify_modality(os.path.join(path, 'coreg_with_' + self.coreg_reference))

        self.mr_shape = nib.load(self.file_paths['t2s']).get_data().shape

        # Mask unused data using t2s ref
        try:
            self.apply_mask_on_mri('t2s')
        except ValueError:
            print(Fore.RED + 'Seemingly, mask is already applied' + Fore.RESET)

    def __repr__(self):
        return '*** Preprocessed Brain Slice *** \n' + 'is_coregistered : ' + str(self.is_coregistered) + '\n' +\
               'reference for coreg :' + str(self.coreg_reference) + '\n' \
               + super(PreprocessedBrainSlice, self).__repr__()

    @staticmethod
    def check_coregistration(path_to_data):
        # Default value if no registration is found
        is_coregistered = False
        coreg_ref = None

        folders_coreg = [os.path.join(path_to_data, f)
                         for f in os.listdir(path_to_data)
                         if os.path.isdir(os.path.join(path_to_data, f)) and f.startswith('coreg_with')]

        if len(folders_coreg) > 0:
            folder_chosen = folders_coreg[0]
            is_coregistered = True
            coreg_ref = folder_chosen.split('coreg_with_')[-1]

        return is_coregistered, coreg_ref

    def coregister_all(self, ref):
        if ref not in self.file_paths.keys():
            raise KeyError(ref + " file does not exists in self.files of mouse " + self.name)

        if os.path.exists(os.path.join(self.path_to_data, "coreg_with_" + ref)):
            print('Coregistration of MR images and " + ref + " already computed !')
        else:
            for modality in self.file_paths:
                self.file_paths[modality] = coregister(self.path_to_data,
                                                       self.file_paths[ref],
                                                       self.file_paths[modality], ref)

    def normalize_all(self, threshold):
        for modality in self.file_paths:
            self.file_paths[modality] = normalize_vol(self.path_to_data, self.file_paths[modality], threshold)

    def apply_mask_on_mri(self, ref):
        mask_outlyers = get_mask_data(self.file_paths[ref], display=False)
        for modality in self.file_paths:
            nifti = nib.load(self.file_paths[modality])
            data = nifti.get_data()
            data[np.invert(mask_outlyers)] = np.nan
            masked_nifti = nib.Nifti1Image(data, affine=nifti.affine, header=nifti.header)
            nib.save(masked_nifti, self.file_paths[modality])


if __name__ == '__main__':
    tg03_preprocessed = PreprocessedBrainSlice('/Users/arnaud.marcoux/histo_mri/images/TG03')
