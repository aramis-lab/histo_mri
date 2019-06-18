import os
import nibabel as nib

class BrainSlice:
    """
    List of attributes :
    name :          name of the brain slice
    file_paths :    dictionary representing the different modalities
    path_to_data :  path to where the data are stored
    histo_path :    path to the histological cut
    """
    def __init__(self, path_to_data):
        from utils import identify_modality

        assert isinstance(path_to_data, str), 'Argument given in parameter must be a valid path'
        self.path_to_data = path_to_data
        self.name = os.path.basename(path_to_data)
        self.file_paths = identify_modality(path_to_data)
        histo_img = [os.path.join(self.path_to_data, f)
                     for f in os.listdir(self.path_to_data)
                     if os.path.basename(f).lower().startswith('histo')]
        if len(histo_img) != 1:
            raise IOError(str(len(histo_img)) + ' image(s) histological images found for ' + self.name)
        self.histo_path = histo_img[0]
        self.mr_dtype = nib.load(self.file_paths['t2s']).get_data().dtype

    def __repr__(self):
        s = (' ** Raw Brain Slice ' + self.name + ' **\n')
        for key in self.file_paths:
            s = s + key + ' ' * (8 - len(key)) + ':    ' + self.file_paths[key] + '\n'
        return s


if __name__ == '__main__':
    tg03 = BrainSlice('/Users/arnaud.marcoux/histo_mri/images/TG03')
