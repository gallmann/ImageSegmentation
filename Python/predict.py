# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 13:22:42 2020

@author: johan
"""


import train
import constants
import utils
import os
import numpy as np
import matplotlib.pyplot as plt
import progressbar
import gdal

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import tensorflow as tf
tf.enable_eager_execution()




def TileGenerator(image_path, tile_size = 256, overlap = 0):
    
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
            
            result*=1./255
            yield result
            
            
            currentx += tile_size-overlap
        currenty += tile_size-overlap
        currentx = 0



def MyImageDataGenerator(image_path,tile_size=256, batch_size = constants.batch_size, overlap = 0):
    '''Image data generator
        Inputs: 
            batch_size - number of images to import at a time
        Output: Decoded RGB image (height x width x 3) 
    '''
    
    tile_generator = TileGenerator(image_path,tile_size=tile_size, overlap = overlap)
    batch = np.empty((batch_size,tile_size,tile_size,3),dtype=np.uint8)
    stop = False
    iv = 0
    while True:
        for i in range(0,batch_size):
            try:
                iv += 1
                batch[i] = next(tile_generator)
            except StopIteration:
                stop = True
        yield batch
        if stop:
            break

    

def save_array_as_image(image_path,image_array):
    
    image_array = image_array.astype(np.uint8)
    if not image_path.endswith(".png") and not image_path.endswith(".jpg") and not image_path.endswith(".tif"):
        print("Error! image_path has to end with .png, .jpg or .tif")
    height = image_array.shape[0]
    width = image_array.shape[1]
    if height*width < Image.MAX_IMAGE_PIXELS:
        newIm = Image.fromarray(image_array, "RGB")
        newIm.save(image_path)
    


def save_array_as_image_with_geo_coords(dst_image_path, image_with_coords, image_array):
    
    fileformat = "GTiff"
    driver = gdal.GetDriverByName(fileformat)
    src_ds = gdal.Open(image_with_coords)
    dst_ds = driver.CreateCopy(dst_image_path, src_ds, strict=0)
    image_array = np.swapaxes(image_array,2,1)
    image_array = np.swapaxes(image_array,1,0)
    dst_ds.GetRasterBand(1).WriteArray(image_array[0], 0, 0)
    dst_ds.GetRasterBand(2).WriteArray(image_array[1], 0, 0)
    dst_ds.GetRasterBand(3).WriteArray(image_array[2], 0, 0)
    dst_ds.FlushCache()  # Write to disk.    


def reassemble(original_image_path, dst_image_path, predictions, classes, tile_size = 256, overlap = 0):
    gdal_image = gdal.Open(original_image_path)
    width = gdal_image.RasterXSize
    height = gdal_image.RasterYSize
    
    mask = np.empty((height,width,3),dtype=np.uint8)
    
    i = 0
    
    currentx = 0
    currenty = 0
    while currenty < height:
        while currentx < width:       
            
            tile = train.onehot_to_rgb(predictions[i],classes)
            mask[currenty:currenty+tile_size,currentx:currentx+tile_size,:3] = tile
            i+=1
            currentx += tile_size-overlap
        currenty += tile_size-overlap
        currentx = 0

    save_array_as_image_with_geo_coords(dst_image_path,original_image_path,mask)
    



def run(working_dir=constants.working_dir, batch_size = constants.batch_size):

    prediction_folder = "C:/Users/johan/Desktop/src_folder2/images"

    
    classes = utils.load_obj(os.path.join(working_dir,"labelmap.pkl"))
    model = train.get_small_unet(n_filters = 32,num_classes=len(classes),batch_size=batch_size)
    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=[train.tversky_loss,train.dice_coef,'accuracy'])
    #model.summary()
    
    
    model_save_path = os.path.join(working_dir,"trained_model.h5")

    model.load_weights(model_save_path)
    
    for image_path in progressbar.progressbar(utils.get_all_image_paths_in_folder(prediction_folder)):
        
        data_generator = MyImageDataGenerator(image_path,batch_size=batch_size)
        
        pred = model.predict_generator(data_generator,steps=22)
        mask_path = os.path.join("C:/Users/johan/Desktop/test", os.path.basename(image_path))
        reassemble(image_path,mask_path,pred,classes)
        
    

run()