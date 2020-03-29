# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:00:59 2020

@author: johan
"""

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

import constants
import utils
import numpy as np
import matplotlib.pyplot as plt
import re
from PIL import Image
import shutil
import progressbar

import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import *


#from tensorflow.keras.engine import Layer
from tensorflow.keras.applications.vgg16 import *
from tensorflow.keras.models import *
#from tensorflow.keras.applications.imagenet_utils import _obtain_input_shape
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Cropping2D, Conv2D
from tensorflow.keras.layers import Input, Add, Dropout, Permute, add
from tensorflow.compat.v1.layers import conv2d_transpose
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
tf.enable_eager_execution()
print("Tensorflow version: " + tf.__version__)


def read_images(image_tiles_dir,mask_tiles_dir):
    '''Function to get all image directories, read images and masks in separate tensors
        Inputs: 
            img_dir - file directory
        Outputs 
            frame_tensors, masks_tensors, frame files list, mask files list
    '''
    
    # Get the file names list from provided directory
    frames_list = utils.get_all_image_paths_in_folder(image_tiles_dir)
    masks_list = utils.get_all_image_paths_in_folder(mask_tiles_dir)
        
    print('{} frame files found in the provided directory.'.format(len(frames_list)))
    print('{} mask files found in the provided directory.'.format(len(masks_list)))
    
    
    # Create dataset of tensors
    frame_data = tf.data.Dataset.from_tensor_slices(frames_list)
    masks_data = tf.data.Dataset.from_tensor_slices(masks_list)
    
    # Read images into the tensor dataset
    frame_tensors = frame_data.map(_read_to_tensor)
    masks_tensors = masks_data.map(_read_to_tensor)
    
    print('Completed importing {} frame images from the provided directory.'.format(len(frames_list)))
    print('Completed importing {} mask images from the provided directory.'.format(len(masks_list)))
    
    return frame_tensors, masks_tensors, frames_list, masks_list

    
def _read_to_tensor(fname, output_height=256, output_width=256, normalize_data=False):
    '''Function to read images from given image file path, and provide resized images as tensors
        Inputs: 
            fname - image file path
            output_height - required output image height
            output_width - required output image width
            normalize_data - if True, normalize data to be centered around 0 (mean 0, range 0 to 1)
        Output: Processed image tensors
    '''
    
    # Read the image as a tensor
    img_strings = tf.io.read_file(fname)
    imgs_decoded = tf.image.decode_jpeg(img_strings)
    
    # Resize the image
    output = tf.image.resize(imgs_decoded, [output_height, output_width])
    
    # Normalize if required
    if normalize_data:
        output = (output - 128) / 128
    return output



def make_folders(project_dir):
    training_data_dir = os.path.join(project_dir,"training_data")
    

    folders = ['train_frames/train', 'train_masks/train', 'val_frames/val', 'val_masks/val']
    
    full_folder_paths = []
    
    for folder in folders:
        full_folder_path = os.path.join(training_data_dir,folder)
        os.makedirs(full_folder_path,exist_ok=True)
        utils.delete_folder_contents(full_folder_path)
        full_folder_paths.append(full_folder_path)
    
    return full_folder_paths
    



def split_train_dir(src_dir_images,src_dir_masks, dst_dir_images, dst_dir_masks, splits):
    """Splits all annotated images into training and testing directory

    Parameters:
        src_dir (str): the directory path containing all images and xml annotation files 
        dst_dir (str): path to the test directory where part of the images (and 
                 annotations) will be copied to
        labels (dict): a dict inside of which the flowers are counted
        labels_dst (dict): a dict inside of which the flowers are counted that
            moved to the dst directory
        split_mode (str): If split_mode is "random", the images are split
            randomly into test and train directory. If split_mode is "deterministic",
            the images will be split in the same way every time this script is 
            executed and therefore making different configurations comparable
        input_folders (list): A list of strings containing all input_folders. 
        splits (list): A list of floats between 0 and 1 of the same length as 
            input_folders. Each boolean indicates what portion of the images
            inside the corresponding input folder should be used for testing or validating and
            not for training
        test_dir_full_size (str): path of folder to which all full size original
            images that are moved to the test directory should be copied to.
            (this folder can be used for evaluation after training) (default is None)
    
    Returns:
        None
    """

    images = utils.get_all_image_paths_in_folder(src_dir_images)

    for input_folder_index in range(0,len(splits)):
        
        portion_to_move_to_dst_dir = float(splits[input_folder_index])
        
        images_in_current_folder = []        

        #get all image_paths in current folder
        for image_path in images:
            if "src_dir" + str(input_folder_index) in image_path:
                images_in_current_folder.append(image_path)
        
                
        import random as rand
        #shuffle the images randomly
        rand.shuffle(images_in_current_folder)
        #and move the first few images to the test folder
        for i in range(0,int(len(images_in_current_folder)*portion_to_move_to_dst_dir)):
            dest_image_file = os.path.join(dst_dir_images,os.path.basename(images_in_current_folder[i]))
            shutil.copyfile(images_in_current_folder[i], dest_image_file)
            src_mask_file = os.path.join(src_dir_masks,os.path.basename(images_in_current_folder[i]))
            dst_mask_file = os.path.join(dst_dir_masks,os.path.basename(images_in_current_folder[i]))
            shutil.copyfile(src_mask_file, dst_mask_file)



def rgb_to_onehot(rgb_image,classes):
    '''Function to one hot encode RGB mask labels
        Inputs: 
            rgb_image - image matrix (eg. 256 x 256 x 3 dimension numpy ndarray)
            colormap - dictionary of color to label id
        Output: One hot encoded image of dimensions (height x width x num_classes) where num_classes = len(colormap)
    '''
    num_classes = len(classes)
    shape = rgb_image.shape[:2]+(num_classes,)
    encoded_image = np.zeros( shape, dtype=np.int8 )
    for i, cls in enumerate(classes):
        encoded_image[:,:,i] = np.all(rgb_image.reshape( (-1,3) ) == utils.id2color(classes,i), axis=1).reshape(shape[:2])
    return encoded_image

def onehot_to_rgb(onehot,classes):
    '''Function to decode encoded mask labels
        Inputs: 
            onehot - one hot encoded image matrix (height x width x num_classes)
            colormap - dictionary of color to label id
        Output: Decoded RGB image (height x width x 3) 
    '''
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) )
    for c in classes:
        output[single_layer==utils.name2id(classes,c)] = utils.name2color(classes,c)
    return np.uint8(output)



def TrainAugmentGenerator(train_frames_dir,train_masks_dir,classes,seed = 1, batch_size = 5):
    '''Train Image data generator
        Inputs: 
            seed - seed provided to the flow_from_directory function to ensure aligned data flow
            batch_size - number of images to import at a time
        Output: Decoded RGB image (height x width x 3) 
    '''
    
    
        # Normalizing only frame images, since masks contain label info
    data_gen_args = dict(rescale=1./255)
    mask_gen_args = dict()
    
    train_frames_datagen = ImageDataGenerator(**data_gen_args)
    train_masks_datagen = ImageDataGenerator(**mask_gen_args)
    
    train_image_generator = train_frames_datagen.flow_from_directory(
    os.path.dirname(train_frames_dir),
    batch_size = batch_size, seed = seed)

    train_mask_generator = train_masks_datagen.flow_from_directory(
    os.path.dirname(train_masks_dir),
    batch_size = batch_size, seed = seed)

    while True:
        X1i = train_image_generator.next()
        X2i = train_mask_generator.next()
        
        #One hot encoding RGB images
        mask_encoded = [rgb_to_onehot(X2i[0][x,:,:,:], classes) for x in range(X2i[0].shape[0])]
        
        yield X1i[0], np.asarray(mask_encoded)


def ValAugmentGenerator(val_frames_dir,val_masks_dir,classes, seed = 1, batch_size = 5):
    '''Validation Image data generator
        Inputs: 
            seed - seed provided to the flow_from_directory function to ensure aligned data flow
            batch_size - number of images to import at a time
        Output: Decoded RGB image (height x width x 3) 
    '''
    # Normalizing only frame images, since masks contain label info
    data_gen_args = dict(rescale=1./255)
    mask_gen_args = dict()

    val_frames_datagen = ImageDataGenerator(**data_gen_args)
    val_masks_datagen = ImageDataGenerator(**mask_gen_args)

    val_image_generator = val_frames_datagen.flow_from_directory(
    os.path.dirname(val_frames_dir),
    batch_size = batch_size, seed = seed)


    val_mask_generator = val_masks_datagen.flow_from_directory(
    os.path.dirname(val_masks_dir),
    batch_size = batch_size, seed = seed)


    while True:
        X1i = val_image_generator.next()
        X2i = val_mask_generator.next()
        
        #One hot encoding RGB images
        mask_encoded = [rgb_to_onehot(X2i[0][x,:,:,:], classes) for x in range(X2i[0].shape[0])]
        
        yield X1i[0], np.asarray(mask_encoded)



def get_small_unet(n_filters = 16, bn = True, dilation_rate = 1, num_classes=20, batch_size=5):
    '''Validation Image data generator
        Inputs: 
            n_filters - base convolution filters
            bn - flag to set batch normalization
            dilation_rate - convolution dilation rate
        Output: Unet keras Model
    '''
    #Define input batch shape
    inputs = Input(batch_shape=(batch_size, 256, 256, 3))
    
    
    conv1 = Conv2D(n_filters * 1, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(inputs)
    if bn:
        conv1 = BatchNormalization()(conv1)
        
    conv1 = Conv2D(n_filters * 1, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv1)
    if bn:
        conv1 = BatchNormalization()(conv1)
    
    pool1 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv1)

    conv2 = Conv2D(n_filters * 2, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(pool1)
    if bn:
        conv2 = BatchNormalization()(conv2)
        
    conv2 = Conv2D(n_filters * 2, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv2)
    if bn:
        conv2 = BatchNormalization()(conv2)
    
    pool2 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv2)

    conv3 = Conv2D(n_filters * 4, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(pool2)
    if bn:
        conv3 = BatchNormalization()(conv3)
        
    conv3 = Conv2D(n_filters * 4, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv3)
    if bn:
        conv3 = BatchNormalization()(conv3)
        
    pool3 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv3)

    conv4 = Conv2D(n_filters * 8, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(pool3)
    if bn:
        conv4 = BatchNormalization()(conv4)
        
    conv4 = Conv2D(n_filters * 8, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv4)
    if bn:
        conv4 = BatchNormalization()(conv4)
        
    pool4 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv4)

    conv5 = Conv2D(n_filters * 16, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(pool4)
    if bn:
        conv5 = BatchNormalization()(conv5)
        
    conv5 = Conv2D(n_filters * 16, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv5)
    if bn:
        conv5 = BatchNormalization()(conv5)
        
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    
    conv6 = Conv2D(n_filters * 8, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(up6)
    if bn:
        conv6 = BatchNormalization()(conv6)
        
    conv6 = Conv2D(n_filters * 8, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv6)
    if bn:
        conv6 = BatchNormalization()(conv6)
        
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    
    conv7 = Conv2D(n_filters * 4, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(up7)
    if bn:
        conv7 = BatchNormalization()(conv7)
        
    conv7 = Conv2D(n_filters * 4, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv7)
    if bn:
        conv7 = BatchNormalization()(conv7)
        
    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    
    conv8 = Conv2D(n_filters * 2, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(up8)
    if bn:
        conv8 = BatchNormalization()(conv8)
        
    conv8 = Conv2D(n_filters * 2, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv8)
    if bn:
        conv8 = BatchNormalization()(conv8)
        
    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    
    conv9 = Conv2D(n_filters * 1, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(up9)
    if bn:
        conv9 = BatchNormalization()(conv9)
        
    conv9 = Conv2D(n_filters * 1, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv9)
    if bn:
        conv9 = BatchNormalization()(conv9)
        
    conv10 = Conv2D(num_classes, (1, 1), activation='softmax', padding = 'same', dilation_rate = dilation_rate)(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    
    return model


# Ref: salehi17, "Twersky loss function for image segmentation using 3D FCDN"
# -> the score is computed for each class separately and then summed
# alpha=beta=0.5 : dice coefficient
# alpha=beta=1   : tanimoto coefficient (also known as jaccard)
# alpha+beta=1   : produces set of F*-scores
# implemented by E. Moebel, 06/04/18
def tversky_loss(y_true, y_pred):
    alpha = 0.5
    beta  = 0.5
    
    ones = K.ones(K.shape(y_true))
    p0 = y_pred      # proba that voxels are class i
    p1 = ones-y_pred # proba that voxels are not class i
    g0 = y_true
    g1 = ones-y_true
    
    num = K.sum(p0*g0, (0,1,2,3))
    den = num + alpha*K.sum(p0*g1,(0,1,2,3)) + beta*K.sum(p1*g0,(0,1,2,3))
    
    T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]
    
    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl-T


def dice_coef(y_true, y_pred, smooth = 1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)



def run(working_dir=constants.working_dir, splits=constants.splits, batch_size=constants.batch_size):
    x = tf.random.uniform([3, 3])

    print("Is there a GPU available: "),
    print(tf.test.is_gpu_available())
    
    print("Is the Tensor on GPU #0:  "),
    print(x.device.endswith('GPU:0'))
    
    print("Device name: {}".format((x.device)))
    
    print("Tensorflow eager execution: " + str(tf.executing_eagerly()))
    
    training_data_dir = os.path.join(working_dir,"training_data")
    mask_tiles_dir = os.path.join(training_data_dir,"masks")
    image_tiles_dir = os.path.join(training_data_dir,"images")


    frame_tensors, masks_tensors, frames_list, masks_list = read_images(image_tiles_dir,mask_tiles_dir)
    
    [train_frames_dir,train_masks_dir,val_frames_dir,val_masks_dir] = make_folders(working_dir)


    
    split_train_dir(image_tiles_dir,mask_tiles_dir,val_frames_dir,val_masks_dir,splits)
    for i in range(len(splits)):
        splits[i] = 1
    split_train_dir(image_tiles_dir,mask_tiles_dir,train_frames_dir,train_masks_dir,splits)
    
    classes = utils.load_obj(os.path.join(working_dir,"labelmap.pkl"))
        
    model = get_small_unet(n_filters = 32,num_classes=len(classes),batch_size=batch_size)

    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=[tversky_loss,dice_coef,'accuracy'])

    model.summary()
    #model.load_weights("model_100_epochs.h5")
    
    model_save_path = os.path.join(working_dir,"trained_model.h5")

    tb = TensorBoard(log_dir=os.path.join(working_dir,"logs"), write_graph=True)
    mc = ModelCheckpoint(mode='max', filepath=model_save_path, monitor='val_acc', save_best_only='True', save_weights_only='True', verbose=1)
    es = EarlyStopping(mode='max', monitor='val_acc', patience=10, verbose=1)
    callbacks = [tb, mc, es]
    
    steps_per_epoch = np.ceil(float(len(frames_list) - round(0.1*len(frames_list))) / float(batch_size))
    validation_steps = np.ceil(float((round(0.1*len(frames_list)))) / float(batch_size))
    
    num_epochs = 100
        
    result = model.fit_generator(TrainAugmentGenerator(train_frames_dir,train_masks_dir,classes,batch_size=batch_size), steps_per_epoch=int(steps_per_epoch) ,
                    validation_data = ValAugmentGenerator(val_frames_dir,val_masks_dir,classes,batch_size=batch_size), 
                    validation_steps = int(validation_steps), epochs=num_epochs, callbacks=callbacks)
    model.save_weights(model_save_path, overwrite=True)
    
    
run()