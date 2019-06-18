from input.preprocessed_brain_slice import PreprocessedBrainSlice
from algo_utils import register_histo, compute_nmi
import numpy as np
import nibabel as nib
from PIL import Image
from os.path import join, exists, basename, dirname
from os import listdir, makedirs
from matplotlib import pyplot as plt
from skimage.transform import SimilarityTransform
from skimage.transform import warp
from scipy.misc import imsave
import os


class InterModalityMatching:

    def __init__(self, preprocessed, create_new_transformation=True):
        assert isinstance(preprocessed, PreprocessedBrainSlice), 'You must provide a PreprocessedBrainSlice'
        self.histo_img = np.array(Image.open(preprocessed.histo_path))
        if create_new_transformation or not exists(join(preprocessed.path_to_data, 'similarity_transform')):
            self.transformation_matrix, histo_img_warped = register_histo(self.histo_img,
                                                                          preprocessed,
                                                                          ref='t1',
                                                                          n_points=7)
            self.save_transformation(preprocessed)
        else:
            self.transformation_matrix = self.choose_best_transformation(join(preprocessed.path_to_data,
                                                                              'similarity_transform'))

        self.display_best_coreg(preprocessed)

    @staticmethod
    def choose_best_transformation(similarity_folder):

        files = [join(similarity_folder, f) for f in listdir(similarity_folder) if f.endswith('txt')]
        best_nmi = 0
        best_transform = -1
        for f in files:
            with open(f) as txtfile:
                nmi_score = 0
                for l in txtfile:
                    if l.startswith('#'):
                        nmi_score = np.float(l[1:])
                if nmi_score > best_nmi:
                    best_nmi = nmi_score
                    best_transform = f
        print("Best similarity transformation for " + basename(dirname(similarity_folder)) + " is : " + best_transform
              + " with score " + str(best_nmi))
        return np.loadtxt(best_transform, delimiter=',')

    def save_transformation(self, preprocessed_brain):
        output_path = join(preprocessed_brain.path_to_data, 'similarity_transform')
        if not exists(output_path):
            makedirs(output_path)
        n_iter = len([f for f in listdir(output_path) if not f.startswith('.')])
        output_file = join(output_path, str(n_iter) + '_similarity_transform.txt')
        np.savetxt(output_file,
                   self.transformation_matrix,
                   delimiter=',',
                   header=str(compute_nmi(preprocessed_brain, self.transformation_matrix)))
        print('File ' + output_file + ' sucessfully saved !')

    # To check implement
    @staticmethod
    def display_best_coreg(brain_slice):

        assert isinstance(brain_slice, PreprocessedBrainSlice), 'Object given in arg is not a a PreprocessedBrainSlice'
        img_folder = os.path.join(brain_slice.path_to_data, 'similarity_transform', 'img')
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)
        sum_of_mr = np.zeros(brain_slice.mr_shape, dtype=brain_slice.mr_dtype)
        for modality in brain_slice.file_paths:
            sum_of_mr += nib.load(brain_slice.file_paths[modality]).get_data()
        sum_of_mr[sum_of_mr != sum_of_mr] = 0
        imsave(os.path.join(img_folder, 'sum_of_mr.png'), sum_of_mr)

        cols = 3
        myfig = plt.figure(5)
        grayscale_histo = np.mean(np.array(Image.open(brain_slice.histo_path)), axis=2)
        myfig.suptitle('Coregistration estimation - choose best')
        n_img = len([f for f in listdir(join(brain_slice.path_to_data, 'similarity_transform')) if not f.startswith('.')])
        rows = np.int(np.ceil(n_img/cols))
        matrix_files = [join(brain_slice.path_to_data, 'similarity_transform', f)
                        for f in listdir(join(brain_slice.path_to_data, 'similarity_transform'))
                        if f.endswith('txt')]
        for n, mat in enumerate(matrix_files):
            sim_matrix = np.loadtxt(mat, delimiter=',')
            sim_transf = SimilarityTransform(matrix=sim_matrix)
            w_histo = warp(grayscale_histo, sim_transf, output_shape=brain_slice.mr_shape, order=1)
            imsave(os.path.join(img_folder, str(n) + '.png'), w_histo)
            nmi_score = np.nan
            with open(mat) as files:
                for line in files:
                    if line.startswith('#'):
                        nmi_score = float(line[1:])
            plt.subplot(rows, cols, n + 1)
            plt.imshow(w_histo + nib.load(brain_slice.file_paths['t1']).get_data())
            plt.axis('off')
            plt.title(basename(mat) + " has mni score : {0:.5f}".format(nmi_score), fontsize=9)
        plt.show()


if __name__ == '__main__':
    tg03 = PreprocessedBrainSlice('/Users/arnaud.marcoux/histo_mri/images/TG03')
    realignment = InterModalityMatching(tg03, create_new_transformation=False)
