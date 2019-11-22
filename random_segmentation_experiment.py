from os.path import join, isfile
from validation.whole_process import WholeProcess
from algorithms.algo_utils import save_object

if __name__ == '__main__':
    # Var
    input_folder = '/localdrive10TB/data/arnaud.marcoux/histo_irm/images'
    output_folder = '/localdrive10TB/data/arnaud.marcoux/histo_irm/output'
    patch_shape = (8, 8)
    random_seg_folder = '/localdrive10TB/data/arnaud.marcoux/histo_irm/random_segmentation'

    for i in range(15):
        random_segmentation = {'TG03': join(random_seg_folder, 'TG03', 'label_tg03_' + str(i) + '.npy'),
                                'TG04': join(random_seg_folder, 'TG04', 'label_tg04_' + str(i) + '.npy'),
                                'TG05': join(random_seg_folder, 'TG05', 'label_tg05_' + str(i) + '.npy'),
                                'TG06': join(random_seg_folder, 'TG06', 'label_tg06_' + str(i) + '.npy'),
                                'WT03': None,
                                'WT04': None,
                                'WT05': None,
                                'WT06': None}
        for key in random_segmentation.keys():
            assert isfile(random_segmentation[key]), random_segmentation[key] + ' does not exists'

    for i in range(15):
        random_segmentation = {'TG03': join(random_seg_folder, 'TG03', 'label_tg03_' + str(i) + '.npy'),
                                'TG04': join(random_seg_folder, 'TG04', 'label_tg04_' + str(i) + '.npy'),
                                'TG05': join(random_seg_folder, 'TG05', 'label_tg05_' + str(i) + '.npy'),
                                'TG06': join(random_seg_folder, 'TG06', 'label_tg06_' + str(i) + '.npy'),
                                'WT03': None,
                                'WT04': None,
                                'WT05': None,
                                'WT06': None}

        out_foldr = join(output_folder, 'random_exp_' + str(i))

        random_process = WholeProcess(input_folder=input_folder,
                                      output_folder=out_foldr,
                                      patch_shape=patch_shape,
                                      segmentation_path=random_segmentation)
        save_object(random_process, join(out_foldr, 'whole_process_' + str(i)))
