import numpy as np
from os.path import isfile


class GridSearch:

    def __init__(self, k_folds, max_epoch, learning_rate, batch_size, hyperparameter_matrix_path):

        for element in [learning_rate, batch_size]:
            assert isinstance(element, list), 'A list of float for learning_rate and a list of int for batch_size must be provided'
        for element in [k_folds, max_epoch]:
            assert isinstance(element, int), 'An int must be provided for n_folds and max_epoch'
        assert isinstance(hyperparameter_matrix_path, str), 'You must provide path to the .npy file that contains (or will contain) the hyperparameter matrix'

        self.n_folds = k_folds
        self.max_epoch = max_epoch
        self.learning_rate_range = learning_rate
        self.batch_size_range = batch_size

        if isfile(hyperparameter_matrix_path):
            self.hyperparameter_matrix = self.load_hyperameters(hyperparameter_matrix_path)
            print('Size of hyperparameter matrix : ' + str(self.hyperparameter_matrix.shape))
            assert self.hyperparameter_matrix.shape == (k_folds, len(learning_rate), len(batch_size), max_epoch), 'Size of hyperparameter matrix does not match size of given parameter range'

            if np.sum(self.hyperparameter_matrix != self.hyperparameter_matrix) == 0:
                self.finished_computation = True
            else:
                self.finished_computation = False
        else:
            self.finished_computation = False
            self.hyperparameter_matrix = np.zeros((k_folds,
                                                   len(learning_rate),
                                                   len(batch_size),
                                                   max_epoch), dtype=np.float)
            self.hyperparameter_matrix[:] = np.nan

        # 4 stands for the number of args to hold in the matrix
        self.hyperparameter_args = np.zeros((k_folds, len(learning_rate), len(batch_size), 3))
        for i, kf in enumerate(k_folds):
            for j, lr in enumerate(self.learning_rate_range):
                for k, bs in enumerate(self.batch_size_range):
                    for l, n in enumerate(range(max_epoch)):
                        self.hyperparameter_args[i, j, k, l] = np.array([kf, lr, bs, n])

    def fit(self, data, target, cnn):
        if self.finished_computation:
            return self.hyperparameter_matrix

        # unfinished business
        non_computed_hyperparameters = np.zeros((self.n_folds, len(self.learning_rate_range), len(self.batch_size_range)))
        for i, kf in enumerate(self.n_folds):
            for j, lr in enumerate(self.learning_rate_range):
                for k, bs in enumerate(self.batch_size_range):
                    non_computed_hyperparameters[i, j, k] = np.sum(self.hyperparameter_matrix[i, j, k, :] != np.sum(self.hyperparameter_matrix[i, j, k, :])) > 0
        idx_non_computed = np.where(non_computed_hyperparameters != non_computed_hyperparameters)
        n_untrained_nn = idx_non_computed[0].shape[0]
        for i in range(n_untrained_nn):
            parameters = self.hyperparameter_args[idx_non_computed[0][i], idx_non_computed[1][i], idx_non_computed[2][i], idx_non_computed[3][i]]
            current_fold = parameters[0]
            current_lr = parameters[1]
            current_bs =parameters[2]
            current_max_epoch = self.n_folds
            self.hyperparameter_matrix = cnn.train_nn()





    @staticmethod
    def save_hyperparameters(nparray, output_path):
        assert isinstance(output_path, str)
        assert isinstance(nparray, np.ndarray)
        np.save(output_path, nparray)

    @staticmethod
    def load_hyperameters(path):
        assert isinstance(path, str)
        return np.load(path)
