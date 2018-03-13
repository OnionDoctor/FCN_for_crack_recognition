不sho# FCN_for_crack_recognition

## Requirements
- Python 3.x
- Tensorflow >= 1.21
- Numpy
- Scipy, Scikit-image
- Matplotlib

## Content
- ```FCN_DatasetReader.py```: Classes for training dataset and test image reading
- ```FCN_layers.py```: Functions of layers
- ```FCN_model.py```: Model of FCN
- ```FCN_finetune.py```: Main training and test of FCN
- ```data/train/*```: Folder for training dataset, contains subfolder 'image', 'annotation' and  'index.txt'
- ```data/valid/*```: Folder for validing dataset, contains subfolder 'image', 'annotation' and  'index.txt'
- ```logs```: Folder for training logs
- ```checkpoints```: Folder for model parameters
- ```test```: Folder for test images

## Useage
### Test
1. Download pretrained model (https://drive.google.com/open?id=1oX7IO0R_ZkfHwZ_zV4c3_v9_24empwNz) and put into folder ```checkpoints```
2. Put test images into folder ```test```
3. Run ```python FCN_finetune.py --mode=predict --test_dir=test```

### Train and finetune
1. Download vgg19 pretrained parameters into the root folder (https://drive.google.com/open?id=15WMDJbFWw3f1qMbTuDO1k4HQ0hyPB4-6)
2. Prepare your own data or download crack dataset from (https://drive.google.com/open?id=1cplcUBmgHfD82YQTWnn1dssK2Z_xRpjx) If you need to change the training samples or validating sample, you can modify the ```index.txt``` file directly. Then put the data into ```data/train/``` and ```data/valid/``` respectively.
3. Run ```python FCN_finetune.py --mode=finetune --learning_rate=1e-4 --num_of_epoch=20 --batch_size=2```
4. If you would like to check the training process, run ```tensorboard --logdir=logs```, then open ```http://localhost:6006/``` using any web explorer.

Please put 'index.txt' into train or valid folder as follows (The feeding process will follow this order):
```
image//0002.jpg,annotation//0002.png
image//0001.jpg,annotation//0001.png
```

### Skeleton of cracks
Once you have got the predictions of cracks, go to python environment
```
from FCN_CrackAnalysis import CrackAnalyse

analyser = CrackAnalyse('test/001.png')
crack_skeleton = analyser.get_skeleton()
crack_lenth = analyser.get_crack_length()
crack_max_width = analyser.get_crack_max_width()
crack_mean_width = analyser.get_crack_mean_width()
```
Then you can using matplotlib to show the skeleton and print the crack morphological features.

## Results
- normal cracks
![crack_cp_0742.png](https://github.com/OnionDoctor/FCN_for_crack_recognition/blob/master/results/crack_cp_0742.png)

- thin cracks
![crack_cp_0063.png](https://github.com/OnionDoctor/FCN_for_crack_recognition/blob/master/results/crack_cp_0063.png)

- intersected cracks
![crack_cp_0070.png](https://github.com/OnionDoctor/FCN_for_crack_recognition/blob/master/results/crack_cp_0070.png)

- historical(wide) cracks
![crack_cp_0228.png](https://github.com/OnionDoctor/FCN_for_crack_recognition/blob/master/results/crack_cp_0228.png)

- mixed cracks
![crack_cp_0286.png](https://github.com/OnionDoctor/FCN_for_crack_recognition/blob/master/results/crack_cp_0286.png)

- complex cracks
<<<<<<< HEAD
![crack_cp_0619.png](https://github.com/OnionDoctor/FCN_for_crack_recognition/blob/master/results/crack_cp_0619.png)
=======
![crack_cp_0619.png](https://raw.githubusercontent.com/OnionDoctor/FCN_for_crack_recognition/blob/master/results/crack_cp_0619.png)
>>>>>>> dbb22682e5c272bb9b20bbc7d399c95ee0fbd28a
