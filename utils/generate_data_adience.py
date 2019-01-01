from PIL import Image
import numpy as np
from keras.utils import np_utils
import random
import sys
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
sys.path.append('../utils')
from data_augmentation_reid  import ImageGenerator
from datasets import DataManager
from datasets import split_data


if __name__ == '__main__':
    dataset_name = 'market'
    batch_size=32
    input_shape = (224,224,3)
    classes=751
    data_loader = DataManager(dataset_name)
    images_path = data_loader.dataset_path 
    ground_truth_data = data_loader.get_data()
    train_keys, val_keys = split_data(ground_truth_data, 0.2)
    #print ground_truth_data
    f = open('../result/market_ground_truth.txt','w')
    f.write(str(ground_truth_data))
    f.close()

    #np.savetxt('../result/market_ground_truth.txt',ground_truth_data)
    #print len(ground_truth_data)
    #print len(train_keys)
    #print train_keys[0]
    #print ground_truth_data[train_keys[0]]
    #print images_path
    #print ground_truth_data[train_keys[1]]
    #print ground_truth_data[train_keys[0]][0]
    #print ground_truth_data[train_keys[1]][0]
    #print ground_truth_data
    image_generator = ImageGenerator(ground_truth_data, batch_size,
                                 input_shape[:2], 
                                 train_keys, val_keys,classes, None,
                                 path_prefix=images_path,
                                 saturation_var=0.5,
                                brightness_var=0.5,
                                contrast_var=0.5,
                                lighting_std=0.5,
                                horizontal_flip_probability=0.5,
                                vertical_flip_probability=0.5,
                                do_random_crop=False,
                                grayscale=False,
                                zoom_range=[0.75, 1.25],
                                translation_factor=.3)
    train_generator = image_generator.flow(mode='train')
    val_generator = image_generator.flow(mode='val')
    #img_names,targets_1,targets_2 = train_generator.next()
    #print img_names['input_1'].shape #[samples,64,64,1]
    #print targets_1['predictions_gender'].shape #[0,1],[1,0]
    #print targets_2['predictions_age'].shape# [0,1,0,0...]
    img_names,targets = train_generator.next()
    x,y= val_generator.next()
    #print type(img_names['input'])
    #print len(x)
    #print img_names['input'].shape
    #print  targets['label']
    #print img_names['input'].shape
    #print targets['label']

