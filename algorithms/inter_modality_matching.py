from input.preprocessed_brain_slice import PreprocessedBrainSlice
from algo_utils import register_histo, compute_nmi
import numpy as np
from PIL import Image
from os.path import join, exists, basename, dirname
from os import listdir, makedirs


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


if __name__ == '__main__':
    tg03 = PreprocessedBrainSlice('/Users/arnaud.marcoux/histo_mri/images/TG03')
    realignment = InterModalityMatching(tg03, create_new_transformation=True)
