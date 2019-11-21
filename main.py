from validation.whole_process import WholeProcess
from os.path import join

if __name__ == '__main__':

    # Var
    input_folder = '/Users/arnaud.marcoux/histo_mri/images'
    output_folder = '/Users/arnaud.marcoux/histo_mri/pickled_data'
    patch_shape = (8, 8)

    # True segmentation: no need to specify them
    true_seg_path = {'TG03': None,
                     'TG04': None,
                     'TG05': None,
                     'TG06': None,
                     'WT03': None,
                     'WT04': None,
                     'WT05': None,
                     'WT06': None}

    true_seg = WholeProcess(input_folder=input_folder,
                            output_folder=join(output_folder, 'true_seg_experiment'),
                            patch_shape=patch_shape,
                            segmentation_path=true_seg_path)

