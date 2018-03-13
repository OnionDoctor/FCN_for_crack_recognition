import numpy as np
import scipy.misc as misc
import os.path as path
import glob

class DatasetReader():
    def __init__(self, imageset_dir, resize=[224, 224], isShuffle=True):
        print("Initialize Dataset Reader ...")

        self.index_file = path.join(imageset_dir, 'index.txt')
        self.img_files, self.ant_files = read_index(self.index_file, imageset_dir)

        self.isShuffle = isShuffle

        if resize == []:
            self.resize = False
        else:
            self.resize = True
            self.height = resize[0]
            self.width = resize[1]

        self.num = len(self.img_files)

        self.imgs = self._read_images(self.img_files)

        self.ants = self._read_images(self.ant_files)

        # initialize batch offset and epoch
        self.reset_batch_offset()
        self.reset_epoch_count()

    def reset_batch_offset(self):
        self.batch_offset = 0

    def reset_epoch_count(self):
        self.epoch_count = 0

    def _read_images(self, image_files):
        return np.array([self._read_image(img_file) if path.exists(img_file) else print(img_file) for img_file in image_files])

    def _read_image(self, image_file):
        image = misc.imread(image_file)
        if self.resize:
            resize_image = misc.imresize(image, [self.width, self.height], interp='nearest')
        else:
            resize_image = image

        # expand 3-dimension tensor to 4-dimension tensor
        if len(resize_image.shape) == 2:
            resize_image = np.expand_dims(resize_image, axis=2)

        # check fate jpg - rgb
        if image_file[-3:] == 'jpg':
            assert resize_image.shape == (224, 224, 3), print(image_file)

        # check fate jpg - gray
        if image_file[-3:] == 'png':
            assert resize_image.shape == (224, 224, 1), print(image_file)
            resize_image = np.divide(resize_image, 255).astype(int)

        return resize_image

    def next_batch(self, batch_size):
        start = self.batch_offset
        end = start + batch_size
        if end <= self.num:
            self.batch_offset = end
            return self.imgs[start: end], self.ants[start: end]
        else:
            # finish one epoch and reset the batch offset
            self.epoch_count += 1
            self.reset_batch_offset()
            # when an epoch finishes, the sequence is reset
            if self.isShuffle:
                sequence = np.arange(self.num)
                np.random.shuffle(sequence)
                self.imgs = self.imgs[sequence]
                self.ants = self.ants[sequence]

            return self.next_batch(batch_size)

class ImageReader():
    def __init__(self, image_dir):
        self.img_files = glob.glob(path.join(image_dir, '*.jpg'))
        self.save_names = [img_file.replace(".jpg", ".png") for img_file in self.img_files]
        self.num = len(self.img_files)

        self.img_index = 0

    def _read_image(self, image_file):
        image = misc.imread(image_file, mode='RGB')
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        return image

    def next_image(self):
        if self.img_index < self.num:
            image = self._read_image(self.img_files[self.img_index])
            name = self.save_names[self.img_index]
            shape = image.shape
            self.img_index += 1
        else:
            self.img_index = 0
            image, name, shape = self.next_image()
        return image, name, shape[:2]

def read_index(index_file, dataset_dir):
    image_files = []
    annotation_files = []
    with open(index_file, 'r') as file:
        for row in file.readlines():
            image_file, annotation_file = row[:-1].split(',')

            image_files.append(dataset_dir + '/' + image_file)
            annotation_files.append(dataset_dir + '/' + annotation_file)
    return image_files, annotation_files

if __name__ == '__main__':
    # datasetReader = DatasetReader('data/valid')
    # for i in range(60):
    #     a, b = datasetReader.next_batch(10)
    #     print(datasetReader.epoch_count, datasetReader.batch_offset)
    #     print(a.shape, b.shape)
    imagedata = ImageReader('compare_cracknet')
    for i in range(imagedata.num):
        print(imagedata.next_image())
