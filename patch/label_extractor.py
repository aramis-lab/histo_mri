from patch.patch_creator import PatchCreator
from algorithms.algo_utils import load_object
from PIL import Image, ImageDraw
import numpy as np
from os.path import join


class LabelExtractor:

    def __init__(self, patches, brain_slice):
        self.counter = 0




if __name__ == '__main__':
    output_dir = '/Users/arnaud.marcoux/histo_mri/pickled_data/tg03'

    tg03 = load_object(join(output_dir, 'TG03'))
    realignment = load_object(join(output_dir, 'realignment'))
    pt = load_object(join(output_dir, 'patches'))

    labels = LabelExtractor(pt, tg03)

