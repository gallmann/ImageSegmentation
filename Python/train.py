# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:00:59 2020

@author: johan
"""

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

import constants
import random as rand
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
    
    all_image_paths = utils.get_all
    

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




def TileGenerator(image_path, tile_size = 256, overlap = 0, rescale = 1./255):
    
    image_array = utils.get_image_array(image_path)
    height = image_array.shape[0]
    width = image_array.shape[1]        
                
    currentx = 0
    currenty = 0
    while currenty < height:
        while currentx < width:       
            
            cropped_array = image_array[currenty:currenty+tile_size,currentx:currentx+tile_size,:3]
            
            result = np.full((tile_size,tile_size,3),0,dtype=np.float64)
            result[:cropped_array.shape[0],:cropped_array.shape[1]] = cropped_array
                        
            result*=rescale
            yield result
            
            
            currentx += tile_size-overlap
        currenty += tile_size-overlap
        currentx = 0


def get_mask_path_from_image_path(image_path):
    def rreplace(s, old, new, occurrence):
        li = s.rsplit(old, occurrence)
        return new.join(li)
    mask_path = image_path.replace(".tif","_mask.png").replace("images","masks")
    return mask_path
    

def MyImageDataGenerator(image_paths,classes,tile_size=256, batch_size = constants.batch_size, overlap = 0):
    '''Image data generator
        Inputs: 
            batch_size - number of images to import at a time
        Output: Decoded RGB image (height x width x 3) 
    '''
    
    batch = np.empty((batch_size,tile_size,tile_size,3),dtype=np.uint8)
    mask_batch = np.empty((batch_size,tile_size,tile_size,len(classes)),dtype=np.uint8)
    i = 0
    image_index = 0
    
    while True:
        image_path = image_paths[image_index % len(image_paths)]
        mask_path = get_mask_path_from_image_path(image_path)
        tile_generator = TileGenerator(image_path,tile_size=tile_size, overlap = overlap)
        mask_tile_generator = TileGenerator(mask_path,tile_size=tile_size, overlap = overlap,rescale=1)
        num_tiles = get_num_frames([image_path],tile_size = tile_size, overlap = overlap)
        for tile_num in range(0,num_tiles):
            batch[i] = next(tile_generator)
            mask_batch[i] = np.asarray(rgb_to_onehot(next(mask_tile_generator), classes))
        
            i += 1
            if i == batch_size:
                i = 0
                yield batch, mask_batch
        image_index += 1



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


def get_data_sets(images_folder):
    
    import random as rand
    all_image_paths = utils.get_all_image_paths_in_folder(images_folder)
    test_image_paths = []
    val_image_paths = []
    train_image_paths = []
    print(len(all_image_paths))
    for src_dir_index in range(0,len(constants.data_source_folders)):
        images_in_current_folder = []
        for image_path in all_image_paths:
            if "_srcdir" + str(src_dir_index) in image_path:
                images_in_current_folder.append(image_path)
        rand.shuffle(images_in_current_folder)    

        for i,path in enumerate(images_in_current_folder):
            if i < constants.val_splits[src_dir_index]*len(images_in_current_folder):
                val_image_paths.append(path)
            elif i < (constants.val_splits[src_dir_index]+constants.test_splits[src_dir_index])*len(images_in_current_folder):
                test_image_paths.append(path)
            else:
                train_image_paths.append(path)
        
        
    
    return [train_image_paths,val_image_paths,test_image_paths] 


def get_num_frames(image_paths,tile_size=256, overlap = 0):
    
    num_frames = 0
    
    for image_path in image_paths:
        image = Image.open(image_path)
        width, height = image.size
        tiles_x = int(np.ceil(float(width)/float(tile_size-overlap)))
        tiles_y = int(np.ceil(float(height)/float(tile_size-overlap)))
        num_frames += tiles_x*tiles_y
    return num_frames
    

def run(working_dir=constants.working_dir, splits=constants.splits, batch_size=constants.batch_size):
    x = tf.random.uniform([3, 3])

    print("Is there a GPU available: "),
    print(tf.test.is_gpu_available())
    
    print("Is the Tensor on GPU #0:  "),
    print(x.device.endswith('GPU:0'))
    
    print("Device name: {}".format((x.device)))
    
    print("Tensorflow eager execution: " + str(tf.executing_eagerly()))
    
    
    training_data_dir = os.path.join(working_dir,"training_data")
    masks_dir = os.path.join(training_data_dir,"masks")
    images_dir = os.path.join(training_data_dir,"images")
    
    [train_image_paths,val_image_paths,test_image_paths] = get_data_sets(images_dir)
    
    
    
    

    
    #frame_tensors, masks_tensors, frames_list, masks_list = read_images(image_tiles_dir,mask_tiles_dir)
    #[train_frames_dir,train_masks_dir,val_frames_dir,val_masks_dir] = make_folders(working_dir)


    
    #split_train_dir(image_tiles_dir,mask_tiles_dir,val_frames_dir,val_masks_dir,splits)
    #for i in range(len(splits)):
    #    splits[i] = 1-splits[i]
    #split_train_dir(image_tiles_dir,mask_tiles_dir,train_frames_dir,train_masks_dir,splits)
    
    classes = utils.load_obj(os.path.join(working_dir,"labelmap.pkl"))
        
    model = get_small_unet(n_filters = 32,num_classes=len(classes),batch_size=batch_size)

    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=[tversky_loss,dice_coef,'accuracy'])

    #model.summary()
    #model.load_weights("model_100_epochs.h5")
    
    model_save_path = os.path.join(working_dir,"trained_model.h5")

    tb = TensorBoard(log_dir=os.path.join(working_dir,"logs"), write_graph=True)
    mc = ModelCheckpoint(mode='max', filepath=model_save_path, monitor='val_acc', save_best_only='True', save_weights_only='True', verbose=1)
    es = EarlyStopping(mode='max', monitor='acc', patience=10, verbose=1)
    callbacks = [tb, mc, es]
    
    steps_per_epoch = int(np.ceil(float(get_num_frames(train_image_paths)) / batch_size))
    validation_steps = int(np.ceil(float(get_num_frames(val_image_paths)) / batch_size))
    #TODO: IF train_images is length 0: warning!!
    
    num_epochs = 100
    
    train_data_generator = MyImageDataGenerator(train_image_paths,classes)  
    val_data_generator = MyImageDataGenerator(val_image_paths,classes)  
    
    result = model.fit_generator(train_data_generator, steps_per_epoch=steps_per_epoch,
                    validation_data = val_data_generator, 
                    validation_steps = validation_steps, epochs=num_epochs, callbacks=callbacks)
    model.save_weights(model_save_path, overwrite=True)
   
    
    
if __name__ == '__main__':
    run()