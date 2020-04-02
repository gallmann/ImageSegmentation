# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 13:31:52 2020

@author: johan
"""
import constants
import os
import train
import utils
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def DataGeneratorWithMasks(train_frames_dir,train_masks_dir,classes,seed = 1, batch_size = 5):
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
        mask_encoded = [utils.rgb_to_onehot(X2i[0][x,:,:,:], classes) for x in range(X2i[0].shape[0])]
        
        yield X1i[0], np.asarray(mask_encoded)


def get_folders(working_dir):
    
    training_data_dir = os.path.join(working_dir,"training_data")

    test_tiles_dir = os.path.join(training_data_dir,"test_frames/0")
    os.makedirs(test_tiles_dir,exist_ok=True)

    test_masks_dir = os.path.join(training_data_dir,"test_masks/0")
    os.makedirs(test_masks_dir,exist_ok=True)
    
    return [test_tiles_dir,test_masks_dir]


    
def run(working_dir=constants.working_dir, batch_size = constants.batch_size):
    
    [test_tiles_dir,test_masks_dir] = get_folders(working_dir)
    
    classes = utils.load_obj(os.path.join(working_dir,"labelmap.pkl"))

    model = train.get_small_unet(n_filters = 32,num_classes=len(classes),batch_size=batch_size)
    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=[train.tversky_loss,train.dice_coef,'accuracy'])
    #model.summary()
    
    model_save_path = os.path.join(working_dir,"trained_model.h5")

    model.load_weights(model_save_path)
    data_generator = DataGeneratorWithMasks(test_tiles_dir,test_masks_dir,classes,batch_size=batch_size)
    
    num_test_frames = len(utils.get_all_image_paths_in_folder(test_tiles_dir))
    steps = int(np.ceil(float(num_test_frames) / batch_size))

    result = model.evaluate(data_generator, steps=steps)
    print(result)

    
    


if __name__ == '__main__':
    run()