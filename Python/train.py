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
import unet_utils



import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
print("Tensorflow version: " + tf.__version__)




def get_folders(working_dir):
    training_data_dir = os.path.join(working_dir,"training_data")
    
    folders = ['train_frames/0', 'train_masks/0', 'val_frames/0', 'val_masks/0', 'test_frames/0', 'test_masks/0']
    
    full_folder_paths = []
    
    for folder in folders:
        full_folder_path = os.path.join(training_data_dir,folder)
        full_folder_paths.append(full_folder_path)
    
    return full_folder_paths


def run(working_dir=constants.working_dir, batch_size=constants.batch_size):
    x = tf.random.uniform([3, 3])

    print("Is there a GPU available: "),
    print(tf.test.is_gpu_available())
    
    print("Is the Tensor on GPU #0:  "),
    print(x.device.endswith('GPU:0'))
    
    print("Device name: {}".format((x.device)))
    
    print("Tensorflow eager execution: " + str(tf.executing_eagerly()))
        
    
    classes = utils.load_obj(os.path.join(working_dir,"labelmap.pkl"))


    [train_frames_dir,train_masks_dir,val_frames_dir,val_masks_dir,test_frames_dir,test_masks_dir] = get_folders(working_dir)
    
    
    num_train_frames = len(utils.get_all_image_paths_in_folder(train_frames_dir))
    num_val_frames = len(utils.get_all_image_paths_in_folder(val_frames_dir))

        
    model_save_path = os.path.join(working_dir,"trained_model.h5")

    
    model = unet_utils.get_small_unet(n_filters = 32,num_classes=len(classes),batch_size=batch_size)

    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=[unet_utils.tversky_loss,unet_utils.dice_coef,'accuracy'])

    #model.summary()
    #model.load_weights("model_100_epochs.h5")
    

    tb = TensorBoard(log_dir=os.path.join(working_dir,"logs"), write_graph=True)
    mc = ModelCheckpoint(mode='max', filepath=model_save_path, monitor='val_acc', save_best_only='True', save_weights_only='True', verbose=1)
    es = EarlyStopping(mode='max', monitor='val_acc', patience=10, verbose=1)
    callbacks = [tb, mc, es]
    
    
    steps_per_epoch = int(np.ceil(float(num_train_frames) / batch_size))
    validation_steps = int(np.ceil(float(num_val_frames) / batch_size))

    num_epochs = 100
    
    
    result = model.fit_generator(unet_utils.DataGeneratorWithMasks(train_frames_dir,train_masks_dir,classes,batch_size=batch_size), steps_per_epoch=int(steps_per_epoch) ,
                    validation_data = unet_utils.DataGeneratorWithMasks(val_frames_dir,val_masks_dir,classes,batch_size=batch_size), 
                    validation_steps = int(validation_steps), epochs=num_epochs, callbacks=callbacks)
    
    #model.save_weights(model_save_path, overwrite=True)
    
    
if __name__ == '__main__':
    run()