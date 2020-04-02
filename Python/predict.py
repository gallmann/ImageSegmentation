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



def DataGenerator(train_frames_dir,train_masks_dir,classes,seed = 1, batch_size = 5):
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
        #print(train_frames_dir)
        X1i = train_image_generator.next()
        X2i = train_mask_generator.next()
        
        #One hot encoding RGB images
        mask_encoded = [rgb_to_onehot(X2i[0][x,:,:,:], classes) for x in range(X2i[0].shape[0])]
        
        
        #np.set_printoptions(threshold=sys.maxsize)

        #print(X1i[0])
        #print(np.asarray(mask_encoded).dtype)
        #import time
        #time.sleep(30)


        yield X1i[0], np.asarray(mask_encoded)

    

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

    test_frames_dir = "C:/Users/johan/Desktop/working_dir/training_data/train_frames/0"
    test_masks_dir = "C:/Users/johan/Desktop/working_dir/training_data/train_masks/0"
    
    classes = utils.load_obj(os.path.join(working_dir,"labelmap.pkl"))
    print(len(classes))
    model = train.get_small_unet(n_filters = 32,num_classes=len(classes),batch_size=batch_size)
    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=[train.tversky_loss,train.dice_coef,'accuracy'])
    #model.summary()
    
    
    model_save_path = os.path.join(working_dir,"trained_model.h5")

    model.load_weights(model_save_path)
    data_generator = DataGenerator(test_frames_dir,test_masks_dir,classes,batch_size=batch_size)
    
    for j in range(0,100):
        
        #data_generator = MyImageDataGenerator(image_path,batch_size=batch_size)
        batch_img,batch_mask = next(data_generator)
        pred_all= model.predict(batch_img)

        #pred = model.predict_generator(data_generator,steps=64)
        #reassemble(image_path,mask_path,pred,classes)
        for i in range(0,np.shape(batch_img)[0]):
            predicted_mask = onehot_to_rgb(pred_all[i],classes)
            mask = onehot_to_rgb(batch_mask[i],classes)
            img = batch_img[i] *255
            
            mask_pred_path = os.path.join("C:/Users/johan/Desktop/test", str(j) + "_" + str(i) + "_mask_pred.png")
            img_path = os.path.join("C:/Users/johan/Desktop/test", str(j) + "_" + str(i) + "_img.png")
            mask_path = os.path.join("C:/Users/johan/Desktop/test", str(j) + "_" + str(i) + "_mask.png")

            save_array_as_image(mask_path,mask)
            save_array_as_image(img_path, img)
            save_array_as_image(mask_pred_path, predicted_mask)


        
if __name__ == '__main__':
    run()