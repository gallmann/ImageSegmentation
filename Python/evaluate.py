# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 13:31:52 2020

@author: johan
"""
import unet_utils
import constants
import os
import utils
import numpy as np
import progressbar
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd



def get_folders(working_dir):
    
    training_data_dir = os.path.join(working_dir,"training_data")

    test_tiles_dir = os.path.join(training_data_dir,"val_frames/0")
    os.makedirs(test_tiles_dir,exist_ok=True)

    test_masks_dir = os.path.join(training_data_dir,"val_masks/0")
    os.makedirs(test_masks_dir,exist_ok=True)
    
    return [test_tiles_dir,test_masks_dir]


    
def run(working_dir=constants.working_dir, batch_size = constants.batch_size, per_class=True):
    
    [test_tiles_dir,test_masks_dir] = get_folders(working_dir)
    
    classes = utils.load_obj(os.path.join(working_dir,"labelmap.pkl"))
    print(classes)

    model = unet_utils.get_small_unet(n_filters = 32,num_classes=len(classes),batch_size=batch_size)
    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=[unet_utils.tversky_loss,unet_utils.dice_coef,'accuracy'])
    #model.summary()
    
    model_save_path = os.path.join(working_dir,"trained_model.h5")

    model.load_weights(model_save_path)
    data_generator = unet_utils.DataGeneratorWithMasks(test_tiles_dir,test_masks_dir,classes,batch_size=batch_size)
    
    num_test_frames = len(utils.get_all_image_paths_in_folder(test_tiles_dir))
    steps = int(np.ceil(float(num_test_frames) / batch_size))
    
    if not per_class:
        model.evaluate(data_generator, steps=steps)
        return


    all_tile_paths = utils.get_all_image_paths_in_folder(test_tiles_dir)
    num_batches = int(np.ceil(float(len(all_tile_paths)) / float(batch_size)))
    
    
    y = np.empty((256*256*len(all_tile_paths),),dtype=np.int8)
    y_pred = np.empty((256*256*len(all_tile_paths),),dtype=np.int8)
    
    for batch_number in progressbar.progressbar(range(0,num_batches)):

        img_batch,mask=next(data_generator)
        
        pred = model.predict(img_batch)
        
        if (batch_number+1)*batch_size > len(all_tile_paths):
            mask = mask[:len(all_tile_paths)-batch_number*(batch_size),:,:,:]
            pred = pred[:len(all_tile_paths)-batch_number*(batch_size),:,:,:]

    
        #result = model.predict(data_generator, steps=steps)
        #print(mask.shape)
        curr_y_index = 256*256*batch_size*batch_number
        next_y_index = 256*256*batch_size*(batch_number+1)
        
        y[curr_y_index:next_y_index] = np.argmax(mask, axis=3).flatten() # Convert one-hot to index
        y_pred[curr_y_index:next_y_index] = np.argmax(pred,axis=3).flatten()
        
        
    print("Computing per class statistics...")
    print(classification_report(y, y_pred,labels=range(0,len(classes)),target_names=classes))
    
    
    print("Computing per class confusion matrix...")
    conf_matrix = confusion_matrix(y, y_pred,labels=range(0,len(classes)),normalize="true")*100
    np.set_printoptions(precision=1,suppress=True)
    
    pd.set_option('display.max_columns', None)

    df = pd.DataFrame(conf_matrix, columns=classes, index=classes)
    df = df.round(1)
    print(df)


    
    
    
    
    
    
    
    
    
    
    
    
    
    


if __name__ == '__main__':
    run()