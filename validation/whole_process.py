from input.preprocessed_brain_slice import PreprocessedBrainSlice
from algorithms.inter_modality_matching import InterModalityMatching
from patch.patch_creator import PatchCreator
from patch.patch_aggregator import PatchAggregator
from cnn.neural_network import HistoNet
from cnn.cross_validation import CrossValidation
from cnn.majority_voting import MajorityVoting
from os.path import join
from os import listdir, mkdir
from algorithms.algo_utils import save_as_pickled_object


class WholeProcess:

    def __init__(self, input_folder: str,
                 output_folder: str,
                 random_segmentation_path: dict,
                 patch_shape=(8, 8)):

        self.mouse_names = ['TG0' + str(i) for i in [3, 4, 5, 6]] + ['WT0' + str(i) for i in [3, 4, 5, 6]]
        labelized_images = [random_segmentation_path[mn] for mn in self.mouse_names] + [None] * 4
        self.brain_slices = [PreprocessedBrainSlice(join(input_folder, mouse_name)) for mouse_name in self.mouse_names]
        self.realignements = [InterModalityMatching(brain_slice, create_new_transformation=False) for brain_slice in self.brain_slices]
        self.patch_creators = [PatchCreator(brain, real, patch_shape, labelized_img=lab_im) for brain, real, lab_im in zip(self.brain_slices, self.realignements, labelized_images)]
        self.patch_aggregator = PatchAggregator(*self.patch_creators)
        self.cnn = HistoNet()
        self.cross_val = CrossValidation(self.cnn, self.patch_aggregator, join(output_folder, 'hyperparameter_tuning'))
        self.majority_voting = MajorityVoting(self.cross_val.best_hyperparameters,
                                              self.patch_aggregator,
                                              self.patch_creators,
                                              self.realignements,
                                              join(output_folder, 'results_on_test_set'))


if __name__ == '__main__':
    random_seg_folder = '/Users/arnaud.marcoux/histo_mri/pickled_data/random_segmentation'
    better_than_chance_fold = '/Users/arnaud.marcoux/histo_mri/better_than_chance_experiment'
    image_folder = '/Users/arnaud.marcoux/histo_mri/images'
    random_seg_total = {}
    for mn in ['TG0' + str(i) for i in [3, 4, 5, 6]]:
        random_seg_total[mn] = [join(random_seg_folder, mn, f) for f in listdir(join(random_seg_folder, mn)) if f.endswith('.npy')]
    random_seg_per_instance = []
    for i in range(len(random_seg_total['TG03'])):
        random_seg_per_instance.append({**{'TG03': random_seg_total['TG03'][i],
                                           'TG04': join(image_folder, 'TG04', 'label_tg04.npy'),
                                           'TG05': random_seg_total['TG05'][i],
                                           'TG06': random_seg_total['TG06'][i]},
                                        **{'WT0' + str(v): join(image_folder, 'WT0' + str(v), 'label_wt0' + str(v) + '.npy')
                                           for v in [3, 4, 5, 6]}})
    for i, files_dict in enumerate(random_seg_per_instance):
        current_out_fold = join(better_than_chance_fold, 'experience_' + str(i))
        try:
            mkdir(current_out_fold)
        except FileExistsError:
            pass
        current_wp = WholeProcess(input_folder=image_folder,
                                  output_folder=current_out_fold,
                                  random_segmentation_path=files_dict,
                                  patch_shape=(8, 8))
        save_as_pickled_object(current_wp, join(current_out_fold, 'pickeled_wp'))