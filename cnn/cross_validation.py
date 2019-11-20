import numpy as np
from patch.patch_aggregator import PatchAggregator
from algorithms.algo_utils import load_object
from cnn.estimator import CnnClassifier
from sklearn.metrics import classification_report
from colorama import Fore
from sklearn.model_selection import GroupKFold, GridSearchCV
from os.path import dirname, join
from cnn.estimate_full_image import FullImageEstimate

def nested_cross_validation_with_grid_search(K1,
                                             K2,
                                             data_agg,
                                             parameter_grid,
                                             CNN,
                                             output_folder,
                                             realignements,
                                             patch_creators):
    """
    Nested cross validation with grid search using scikit-learn libraries. Our model, the CNN is wrapped into an estimator
    from scikit learn. This allows us to use cross validation procedures.

    :param K1: number of outter folds. In our case, 4
    :param K2: number of inner fold. In our case 3
    :param data_agg: All our data label
    :param parameter_grid: Dictionnaries of paramters that we want to test for our model
    :param CNN: Our estimator
    :param output_file: Our file with results from differents splits, performance and best parameters
    :param realignements: Realignement list
    :param patch_creators: patch creators
    :return: Nothing
    """
    group_kfold_outter = GroupKFold(n_splits=K1)

    X = data_agg.all_patches
    y = data_agg.all_labels

    # Trick to group each time a transgenic slice with a wild type
    reformat_group_name = [elem[-2:] for elem in data_agg.mouse_name]
    reformat_group_name = np.array(reformat_group_name)
    group_names = np.array(data_agg.mouse_name)

    # Split dataset for outter fold
    for i, (train_index_outter, test_index_outter) in enumerate(group_kfold_outter.split(X, y, reformat_group_name)):
        train_slices = list(set(group_names[train_index_outter]))
        test_slices = list(set(group_names[test_index_outter]))

        describe_outter_split = ("GROUP TRAIN: " + str(train_slices) + ' (' + str(len(train_index_outter)) + ') '
                                 + "GROUP TEST: " + str(test_slices) + ' (' + str(len(test_index_outter)) + ')\n')
        print(describe_outter_split)

        group_kfold_inner = GroupKFold(n_splits=K2)

        grid_search_cv = GridSearchCV(CNN,
                                      parameter_grid,
                                      cv=group_kfold_inner.split(X[train_index_outter],
                                                                 y[train_index_outter],
                                                                 reformat_group_name[train_index_outter]),
                                      refit=False)
        grid_search_cv.fit(X, y)
        optimal_parameters = grid_search_cv.best_params_
        best_model_for_this_split = CnnClassifier(learning_rate=optimal_parameters['learning_rate'],
                                                  batch_size=optimal_parameters['batch_size'],
                                                  n_epochs=optimal_parameters['n_epochs'])
        best_model_for_this_split.fit(X[train_index_outter],
                                      y[train_index_outter])
        report = ('Performance on split ' + str(i) + ':\n\tBest parameters:\n'
                  +'\n\tbatch size: ' + str(optimal_parameters['batch_size'])
                  + '\n\tlearning rate: ' + str(optimal_parameters['learning_rate'])
                  + '\n\tn epochs: ' + str(optimal_parameters['n_epochs']) + '\n'
                  + classification_report(y_true=y[test_index_outter],
                                          y_pred=best_model_for_this_split.predict(X[test_index_outter]),
                                          target_names=['no-dnf', 'dnf'],
                                          digits=3))
        print(report)
        final_txt = describe_outter_split + report
        add_log_information(final_txt, join(output_folder, 'report.txt'))

        # Now estimate full images
        for sl in test_slices:
            full_image_estimation = FullImageEstimate(best_model_for_this_split,
                                                      patch_creators[slices_to_idx[sl]],
                                                      realignements[slices_to_idx[sl]],
                                                      (384, 384))
            full_image_estimation.show_estimate(join(output_folder,
                                                     'split-' + str(i) + '_slice-' + sl + '.png'))



def add_log_information(my_txt, file):
    # a for append, and + if it does not exist
    with open(file, 'a+') as f:
        f.write(my_txt)

slices_to_idx = {'TG03': 0,
                 'TG04': 1,
                 'TG05': 2,
                 'TG06': 3,
                 'WT03': 4,
                 'WT04': 5,
                 'WT05': 6,
                 'WT06': 7}

if __name__ == '__main__':
    aggregator = load_object('/Users/arnaud.marcoux/histo_mri/results/patch_aggregator_8_8')
    realignements = load_object('/Users/arnaud.marcoux/histo_mri/results/realignements')
    patch_creators = load_object('/Users/arnaud.marcoux/histo_mri/results/patch_creators_8_8')

    CNN = CnnClassifier()
    parameters = {'n_epochs': [1],
                  'learning_rate': list(np.logspace(-4, -1, 8)),
                  'batch_size': [32, 64]}

    parameters = {'n_epochs': [1],
                  'learning_rate': [0.1],
                  'batch_size': [32]}

    nested_cross_validation_with_grid_search(K1=4,
                                             K2=3,
                                             data_agg=aggregator,
                                             parameter_grid=parameters,
                                             CNN=CNN,
                                             output_folder='/Users/arnaud.marcoux/histo_mri/results/cross_val/true_seg2',
                                             realignements=realignements,
                                             patch_creators=patch_creators)

