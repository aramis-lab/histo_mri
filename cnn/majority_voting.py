from cnn.neural_network import HistoNet
from torch.utils.data.dataloader import DataLoader
from patch.patch_aggregator import PatchAggregator
from patch.patch_creator import PatchCreator
from torch.utils.data.dataset import TensorDataset
from cnn.estimate_full_image import FullImageEstimate
from algorithms.inter_modality_matching import InterModalityMatching
from typing import List
import numpy as np
import os


class MajorityVoting:

    def __init__(self, best_parameters: dict,
                 patch_aggregator: PatchAggregator,
                 patch_creators: List[PatchCreator],
                 realignments: InterModalityMatching,
                 output_dir: str):
        """
        :param best_parameters: dictionnary of best parameters obtained with CrossValidation class
        :param patch_aggregator: patch aggregator
        :param output_dir:
        """
        self.cnn_pool = [HistoNet() for _ in range(3)]
        self.val_balanced_accuracy = self.train_across_folds(best_parameters, patch_aggregator)
        self.test_balanced_accuracy = self.evaluate_on_test_set(patch_aggregator.get_tensor(*best_parameters['test_set']),
                                                                best_parameters['test_set'],
                                                                patch_creators,
                                                                realignments,
                                                                output_dir)

    def train_across_folds(self, best_parameters, patch_aggregator):
        epochs = 10
        balanced_accuracy = np.zeros((len(self.cnn_pool), epochs))
        for i in range(len(self.cnn_pool)):
            current_params = {'batch_size': int(best_parameters['batch_size']),
                              'shuffle': True,
                              'num_workers': 8}
            dataloader_train = DataLoader(patch_aggregator.get_tensor(*best_parameters['train_set'][i]),
                                          **current_params)
            dataset_test = patch_aggregator.get_tensor(*best_parameters['val_set'][i])
            balanced_accuracy[i, :] = self.cnn_pool[i].train_nn(dataloader_train,
                                                                lr=best_parameters['learning_rate'],
                                                                n_epoch=epochs,
                                                                val_data=dataset_test)
        return balanced_accuracy

    def evaluate_on_test_set(self,
                             test_set: TensorDataset,
                             test_set_name: List[str],
                             patch_creator: List[PatchCreator],
                             realignments: List[InterModalityMatching],
                             output_folder: str):

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        balanced_accuracy = []

        idx_test_set = [i for i, elem in enumerate(patch_creator) if elem.name in test_set_name]

        for i, cnn in enumerate(self.cnn_pool):
            balanced_accuracy.append(cnn.test(test_set.tensors[0], test_set.tensors[1]))
            for idx in idx_test_set:
                img_estimation = FullImageEstimate(cnn,
                                                   patch_creator[idx],
                                                   realignments[idx],
                                                   (384, 384))
                img_estimation.show_estimate(os.path.join(output_folder, 'fold_' + str(i)))

        return balanced_accuracy
