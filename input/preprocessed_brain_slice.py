import os
from raw_brain_slice import BrainSlice
from utils import identify_modality, coregister, normalize_vol


class PreprocessedBrainSlice(BrainSlice):
    """
    List of attributes :
    name :          name of the brain slice
    file_paths :    dictionary representing the different modalities
    """

    def __init__(self, path):
        super(PreprocessedBrainSlice, self).__init__(path)
        self.is_coregistered, self.coreg_reference = self.check_coregistration(path)

        if not self.is_coregistered:
            if not os.path.exists(os.path.join(self.path_to_data, 'normalized')):
                self.normalize_all(threshold=500)
                self.file_paths = identify_modality(os.path.join(self.path_to_data, 'normalized'))
            self.coreg_reference = 't2s'
            self.coregister_all(self.coreg_reference)
            self.is_coregistered = True

        self.file_paths = identify_modality(os.path.join(path, 'coreg_with_' + self.coreg_reference))

    def __repr__(self):
        return '*** Preprocessed Brain Slice *** \n' + 'is_coregistered : ' + str(self.is_coregistered) + '\n' +\
               'reference for coreg :' + str(self.coreg_reference) + '\n' + super(PreprocessedBrainSlice, self).__repr__()

    def check_coregistration(self, path_to_data):
        folders_coreg = [os.path.join(path_to_data, f)
                         for f in os.listdir(path_to_data)
                         if os.path.isdir(os.path.join(path_to_data, f)) and f.startswith('coreg_with')]

        if len(folders_coreg) > 0:
            folder_chosen = folders_coreg[0]
            is_coregistered = True
            coreg_ref = folder_chosen.split('coreg_with_')[-1]
        elif len(folders_coreg) == 0:
            is_coregistered = False
            coreg_ref = None

        return is_coregistered, coreg_ref

    def coregister_all(self, ref):
        if ref not in self.file_paths.keys():
            raise KeyError(ref + " file does not exists in self.files of mouse " + self.name)

        if os.path.exists(os.path.join(self.path_to_data, "coreg_with_" + ref)):
            print('Coregistration of MR images and " + ref + " already computed !')
        else:
            for modality in self.file_paths:
                self.file_paths[modality] = coregister(self.path_to_data, self.file_paths[ref], self.file_paths[modality], ref)

    def normalize_all(self, threshold):
        for modality in self.file_paths:
            self.file_paths[modality] = normalize_vol(self.path_to_data, self.file_paths[modality], threshold)


if __name__ == '__main__':
    tg03_preprocessed = PreprocessedBrainSlice('/Users/arnaud.marcoux/histo_mri/images/TG03')
