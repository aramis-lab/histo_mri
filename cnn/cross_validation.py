from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np
from os.path import isfile
import torch
import numpy as np
from patch.patch_aggregator import PatchAggregator
from algorithms.algo_utils import load_object
from cnn.neural_network import HistoNet
from torch.utils.data.dataset import TensorDataset, Subset
from torch.utils.data.dataloader import DataLoader
from os.path import join, isfile, exists
from os import remove, mkdir, getcwd


class CrossValidation:

    def __init__(self, cnn, data_aggreagator, output_folder):
        self.hyperparameter_matrix, self.best_hyperparameters = self.find_hyper_parameter(cnn,
                                                                                          data_aggreagator,
                                                                                          output_directory=output_folder,
                                                                                          n_epochs=15)

    @staticmethod
    def write_informations(n_epoch, lr_range, batch_size_range, train_set, val_set, test_set, patch_shape, in_mat, output_file):
        """
        :param n_epoch:
        :param lr_range:
        :param batch_size_range:
        :param train_set:
        :param val_set:
        :param test_set:
        :param patch_shape:
        :param in_mat:
        :param output_file:
        :return:
        """

        mat = np.load(in_mat)
        # Averaging along folds
        mat_avg = np.mean(mat, axis=0)
        argmax = np.unravel_index(np.argmax(mat_avg), mat_avg.shape)

        best_accuracy_fold = mat[:, argmax[0], argmax[1], :]
        idx_best_fold = np.unravel_index(np.argmax(best_accuracy_fold), best_accuracy_fold.shape)[0]

        if isfile(output_file):
            remove(output_file)

        with open(output_file, 'w') as file:
            file.write('\npatch shape : ' + str(patch_shape))
            file.write('\nn_epoch : ' + str(n_epoch))
            file.write('\nlr_range : ' + str(lr_range))
            file.write('\nbatch_size_range : ' + str(batch_size_range))
            file.write('\nmatrix file : ' + str(in_mat))
            file.write('\ntrain set : ' + str(train_set))
            file.write('\nval set : ' + str(val_set))
            file.write('\ntest set' + str(test_set))
            file.write('\n\n** Best model :\nlr : ' + str(lr_range[argmax[0]]))
            file.write('\nbatch size : ' + str(batch_size_range[argmax[1]]))
            file.write('\nepoch : ' + str(argmax[2]))
            file.write('\nidx best fold : ' + str(idx_best_fold))
            file.write('\nBest balanced accuracy (avg across folds): ' + str(np.max(mat_avg)))

        return {'batch_size': batch_size_range[argmax[1]], 'learning_rate': lr_range[argmax[0]], 'epoch': argmax[2]}

    @staticmethod
    def divide_population():
        n_tg, n_wt = 4, 4
        assert n_tg == n_wt, 'This cross validation procedure is designed to work only when n_tg == n_wt'

        tg = np.arange(0, 4, 1)
        wt = np.arange(0, 4, 1)

        # Real draw
        # test_tg = np.random.choice(tg, 1, replace=False)[0]
        # test_wt = np.random.choice(wt, 1, replace=False)[0]
        test_tg = 1
        test_wt = 3

        tg = [elem for elem in tg if elem != test_tg]
        wt = [elem for elem in wt if elem != test_wt]

        kf_tg = KFold(n_splits=n_tg - 1, shuffle=True)
        kf_wt = KFold(n_splits=n_wt - 1, shuffle=True)

        train_set = []
        val_set = []
        test_set = [test_tg, test_wt]

        for (idx_train_tg, idx_val_tg), (idx_train_wt, idx_val_wt) in zip(kf_tg.split(tg), kf_wt.split(wt)):
            train_set.append(([tg[idx] for idx in idx_train_tg], [wt[idx] for idx in idx_train_wt]))
            val_set.append(([tg[idx] for idx in idx_val_tg], [wt[idx] for idx in idx_val_wt]))

        return train_set, val_set, test_set

    def find_hyper_parameter(self,
                             cnn,
                             data,
                             n_epochs=6,
                             lr_range=np.logspace(-3, -1, 6),
                             batch_size_range=np.array([32, 64]),
                             output_directory=getcwd()):
        """
        :param cnn: Neural net
        :param data: patchAggregator
        :param n_epochs: n_epoch to run while searching for best hyperparameters
        :param lr_range: learning_rate range
        :param batch_size_range: batch_size range
        :param output_directory:
        :return:
        """

        if not exists(output_directory):
            mkdir(output_directory)

        # CV model
        k_fold = 3

        accuracy_hyperparameters = np.zeros((k_fold,
                                             lr_range.size,
                                             batch_size_range.size,
                                             n_epochs)).astype('float')
        # len(train_slices = k_fold (3)
        # len(val_slices = k_fold (3)
        # len (test_slices) = 1
        train_slices, val_slices, test_slices = self.divide_population()
        train_slices_name = []
        val_slices_name = []
        test_slices_name = ['TG0' + str(test_slices[0] + 3), 'WT0' + str(test_slices[1] + 3)]

        for train_slice, val_slice in zip(train_slices, val_slices):
            train_slices_name.append(['TG0' + str(n + 3) for n in train_slice[0]]
                                     + ['WT0' + str(n + 3) for n in train_slice[1]])
            val_slices_name.append(['TG0' + str(n + 3) for n in val_slice[0]]
                                   + ['WT0' + str(n + 3) for n in val_slice[1]])
        train_idx = []
        val_idx = []

        for k, (train_slice_name, val_slice_name) in enumerate(zip(train_slices_name, val_slices_name)):
            train_idx.append([i for i in range(len(data.mouse_name)) if data.mouse_name[i] in train_slice_name])
            val_idx.append([i for i in range(len(data.mouse_name)) if data.mouse_name[i] in val_slice_name])

        dataset = TensorDataset(torch.from_numpy(data.all_patches),
                                torch.from_numpy(data.all_labels))
        best_hyperparameters = {}
        # k_fold grid
        for idx_k in range(k_fold):
            # Those lines displays information of the repartition of dnf labels within each train / val t

            subset_train = Subset(dataset, train_idx[idx_k])

            for idx_lr, lr in enumerate(lr_range):
                for idx_bs, batch_size in enumerate(batch_size_range):
                    current_params = {'batch_size': int(batch_size),
                                      'shuffle': True,
                                      'num_workers': 8}
                    dataloader_train = DataLoader(subset_train, **current_params)
                    dataset_val = data.get_tensor(*val_slices_name[idx_k])
                    accuracy_hyperparameters[idx_k, idx_lr, idx_bs, :] = cnn.train_nn(dataloader_train,
                                                                                      lr=lr,
                                                                                      n_epoch=n_epochs,
                                                                                      val_data=dataset_val)
                    np.save(join(output_directory, 'hyperparameter_matrix.npy'), accuracy_hyperparameters)
                    best_hyperparameters = self.write_informations(n_epochs, lr_range, batch_size_range,
                                                                   train_slices_name,
                                                                   val_slices_name,
                                                                   test_slices_name,
                                                                   data.all_patches[0].shape,
                                                                   join(output_directory, 'hyperparameter_matrix.npy'),
                                                                   join(output_directory, 'results_cnn.txt'))
        return accuracy_hyperparameters, best_hyperparameters


if __name__ == '__main__':
    aggregator = load_object('/Users/arnaud.marcoux/histo_mri/pickled_data/patch_aggregator_8_8')
    CNN = HistoNet()
    cross_val = CrossValidation(cnn=CNN,
                                data_aggreagator=aggregator,
                                output_folder='/Users/arnaud.marcoux/histo_mri/pickled_data/simple_model')
