from input.preprocessed_brain_slice import PreprocessedBrainSlice
from algorithms.inter_modality_matching import InterModalityMatching
from patch.patch_creator import PatchCreator
from patch.patch_aggregator import PatchAggregator
from cnn.neural_network import HistoNet
from os.path import join
import numpy as np
from os import listdir, mkdir
from algorithms.algo_utils import save_as_pickled_object
from cross_validation import nested_cross_validation_with_grid_search
from cnn.estimator import CnnClassifier



class WholeProcess:

    def __init__(self,
                 input_folder: str,
                 output_folder: str,
                 segmentation_path: dict,
                 patch_shape=(8, 8)):


        self.mouse_names = ['TG0' + str(i) for i in [3, 4, 5, 6]] + ['WT0' + str(i) for i in [3, 4, 5, 6]]
        labelized_images = [segmentation_path[mn]
                            for mn in self.mouse_names] + [None] * 4
        self.brain_slices = [PreprocessedBrainSlice(join(input_folder, mouse_name))
                             for mouse_name in self.mouse_names]
        self.realignements = [InterModalityMatching(brain_slice, create_new_transformation=False)
                              for brain_slice in self.brain_slices]
        self.patch_creators = [PatchCreator(brain, real, patch_shape, labelized_img=lab_im)
                               for brain, real, lab_im in zip(self.brain_slices,
                                                              self.realignements,
                                                              labelized_images)]
        self.patch_aggregator = PatchAggregator(*self.patch_creators)
        self.cnn = CnnClassifier()

        self.paramter_grid = {'n_epochs': [8, 12],
                            'learning_rate': list(np.logspace(-4, -1, 8)),
                            'batch_size': [32, 64]}
        nested_cross_validation_with_grid_search(K1=4,
                                                 K2=3,
                                                 data_agg=self.patch_aggregator,
                                                 parameter_grid=self.paramter_grid,
                                                 CNN=self.cnn,
                                                 output_folder=output_folder,
                                                 realignements=self.realignements,
                                                 patch_creators=self.patch_creators)
