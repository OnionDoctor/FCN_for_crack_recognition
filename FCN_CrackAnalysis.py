from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt

from skimage import morphology, feature, measure, exposure, filters, color
from scipy import ndimage as ndi

import scipy.misc as misc

BLUE = '#00b8e6'
DARKBLUE = '#00343f'
RED = '#ff5983'
DARKRED = '#7a023c'
YELLOW = '#ffe957'
DARKYELLOW = '#f29f3f'
GREEN = '#61ff69'
DARKGREEN = '#0b6e48'
GRAY = '#cccccc'

class CrackAnalyse(object):
    def __init__(self, predict_image_file):
        # load
        img = misc.imread(predict_image_file, mode='L')
        img_size = img.size
        self.img = img

        # binary
        img_bnr = (img > 0).astype(np.uint8)

        # opening and closing
        img_bnr = ndi.morphology.binary_closing(img_bnr)
        img_bnr = ndi.morphology.binary_opening(img_bnr)

        self.img_bnr = img_bnr

        # segmentation
        img_labels, num_labels = ndi.label(img_bnr)
        # background label = 0
        labels = range(1, num_labels + 1)
        sizes = ndi.sum(img_bnr, img_labels, labels)

        # argsort according to size descend
        order = np.argsort(sizes)[::-1]
        labels = [labels[i] for i in order]

        img_sgt = img_labels / np.max(labels)
        self.img_sgt = img_sgt

        crack_lens = []
        crack_max_wids = []
        img_skl = np.zeros_like(self.img, dtype=np.float32)

        # skeletonize - median
        for label in labels:
            mask = img_labels == label
            # save the steps for analyse
            # misc.imsave(str(label) + '.png', mask / (num_labels + 1))

            median_axis, median_dist = morphology.medial_axis(img_sgt, mask, return_distance=True)
            crack_len = np.sum(median_axis)
            crack_max_wid = np.max(median_dist)
            img_mph = median_axis * median_dist

            crack_lens.append(crack_len)
            crack_max_wids.append(crack_max_wid)
            img_skl += img_mph

        '''
        # skeleton - zhand and suen
        for label in labels:
            mask = (img_labels == label).astype(np.uint8)
            crack_skl = morphology.skeletonize(mask)
            crack_len = np.sum(crack_skl)
            crack_width = np.sum(mask) / crack_len

            crack_lens.append(crack_len)
            crack_max_wids.append(crack_width)
            img_skl += crack_skl
        '''

        self.img_skl = img_skl
        self.crack_lens = np.array(crack_lens)
        self.crack_max_wids = np.array(crack_max_wids)
        self.ratio = np.sum(img_bnr) / img_size

    def get_prediction(self):
        return self.img

    def get_segmentation(self):
        return self.img_sgt

    def get_skeleton(self):
        return ndi.grey_dilation(self.img_skl, size=2)

    def get_crack_lens(self):
        return self.crack_lens

    def get_crack_wids(self):
        return self.crack_max_wids

    def get_crack_length(self):
        return np.sum(self.crack_lens)

    def get_crack_max_width(self):
        return np.max(self.crack_max_wids)

    def get_crack_mean_width(self):
        return np.sum(self.img_bnr) / np.sum(self.crack_lens)

    def get_ratio(self):
        return self.ratio

class Edge_Detector(object):
    def __init__(self, original_image):
        img_gray = color.rgb2gray(original_image)
        self.img_gray = img_gray

    def get_edges(self, detector='sobel'):
        if detector == 'sobel':
            img = filters.sobel(self.img_gray)
        elif detector == 'canny1':
            img = feature.canny(self.img_gray, sigma=1)
        elif detector == 'canny3':
            img = feature.canny(self.img_gray, sigma=3)
        elif detector == 'scharr':
            img = filters.scharr(self.img_gray)
        elif detector == 'prewitt':
            img = filters.prewitt(self.img_gray)
        elif detector == 'roberts':
            img = filters.roberts(self.img_gray)
        return img

def Hilditch_skeleton(binary_image):
    size = binary_image.size
    skel = np.zeros(binary_image.shape, np.uint8)

    elem = np.array([[0, 1, 0],
                     [1, 1, 1],
                     [0, 1, 0]])

    image = binary_image.copy()
    for _ in range(10000):
        eroded = ndi.binary_erosion(image, elem)
        temp = ndi.binary_dilation(eroded, elem)
        temp = image - temp
        skel = np.bitwise_or(skel, temp)
        image = eroded.copy()

        zeros = size - np.sum(image > 0)
        if zeros == size:
            break

    return skel


if __name__ == '__main__':
    # skeleton compare
    # skeleton_compare('0633', True, 'skl_cp_0633')
    # skeleton_compare('0203', True, 'skl_cp_0203')

    # result compare
    # crack_compare('0742', True, 'crack_cp_0742')
    # for i in range(34, 777):
    #     name = '%04d' % i
    #     crack_compare(name, True, 'test/crack_cp_' + name)
    # crack_compare('0293', False)

    # crack_feature_summary()
    # feature_compare('length')
    # feature_compare('max_width')
    # feature_compare('mean_width')

    # crack_feature_compare('0002', True, 'feature_cp_0002')
    # crack_feature_compare('0223', True, 'feature_cp_0223')
    # crack_feature_compare('0035', True, 'feature_cp_0035')
    # crack_feature_compare('0605', True, 'feature_cp_0605')

    crack_recogniation_compare('001', True, 'recognition_cp_001')
    crack_recogniation_compare('002', True, 'recognition_cp_002')
    crack_recogniation_compare('003', True, 'recognition_cp_003')
    crack_recogniation_compare('004', True, 'recognition_cp_004')