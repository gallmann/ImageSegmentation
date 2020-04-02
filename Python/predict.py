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



def DataGenerator(train_frames_dir,classes,seed = 1, batch_size = 5):
    '''Train Image data generator
        Inputs: 
            seed - seed provided to the flow_from_directory function to ensure aligned data flow
            batch_size - number of images to import at a time
        Output: Decoded RGB image (height x width x 3) 
    '''
    
    
    # Normalizing only frame images, since masks contain label info
    data_gen_args = dict(rescale=1./255)
    
    
    train_frames_datagen = ImageDataGenerator(**data_gen_args)
    
    train_image_generator = train_frames_datagen.flow_from_directory(
    os.path.dirname(train_frames_dir),
    batch_size = batch_size, seed = seed, shuffle = False)
    
    filenames = train_image_generator.filenames
    
    batch_index = 0
    
    while True:   
        
        yield train_image_generator.next()[0],filenames[batch_index:(batch_index+batch_size)%(len(filenames)+1)]
        batch_index += batch_size




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


def reassemble(original_image_path, dst_image_path, pred_tiles_dir, tile_size = 256, overlap = 0):
    
    gdal_image = gdal.Open(original_image_path)
    width = gdal_image.RasterXSize
    height = gdal_image.RasterYSize
        
    original_image_name = os.path.basename(original_image_path).replace("_mask.tif","").replace(".tif","")
    
    result = np.empty((height,width,3),dtype=np.uint8)

    
    currentx = 0
    currenty = 0
    while currenty < height:
        while currentx < width:     
            
            tile_path = os.path.join(pred_tiles_dir,  original_image_name + "_subtile_" + "x" + str(currentx) + "_y" + str(currenty) + ".png")
            tile = utils.get_image_array(tile_path)
            
            result[currenty:currenty+tile_size,currentx:currentx+tile_size,:3] = tile
            currentx += tile_size-overlap
        currenty += tile_size-overlap
        currentx = 0

    save_array_as_image_with_geo_coords(dst_image_path,original_image_path,result)
    

def make_folders(working_dir):
    temp_dir = os.path.join(working_dir,"temp")
    os.makedirs(temp_dir,exist_ok=True)
    utils.delete_folder_contents(temp_dir)
    
    tiles_dir = os.path.join(temp_dir,"tiles/0")
    os.makedirs(tiles_dir,exist_ok=True)

    pred_tiles_dir = os.path.join(temp_dir,"pred_tiles/0")
    os.makedirs(pred_tiles_dir,exist_ok=True)

    
    return [temp_dir, tiles_dir, pred_tiles_dir]


def run(predict_folder,output_folder, working_dir=constants.working_dir, batch_size = constants.batch_size):

    
    [temp_dir, tiles_dir, pred_tiles_dir] = make_folders(working_dir)
    classes = utils.load_obj(os.path.join(working_dir,"labelmap.pkl"))

    print("Preparing image tiles...")
    images_to_predict = utils.get_all_image_paths_in_folder(predict_folder)
    for image_path in progressbar.progressbar(images_to_predict):
        utils.tile_image(image_path, tiles_dir, classes, tile_size=256, overlap=0)
    
    print("Loading Trained Model...")
    
    all_tile_paths = utils.get_all_image_paths_in_folder(tiles_dir)

    #test_frames_dir = "C:/Users/johan/Desktop/working_dir/training_data/train_frames/0"
    #test_masks_dir = "C:/Users/johan/Desktop/working_dir/training_data/train_masks/0"
    
    
    
    #print(len(classes))
    
    model = train.get_small_unet(n_filters = 32,num_classes=len(classes),batch_size=batch_size)
    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=[train.tversky_loss,train.dice_coef,'accuracy'])
    #model.summary()
    
    model_save_path = os.path.join(working_dir,"trained_model.h5")

    model.load_weights(model_save_path)
    data_generator = DataGenerator(tiles_dir,classes,batch_size=batch_size)
    
    num_batches = int(np.ceil(float(len(all_tile_paths)) / float(batch_size)))
    tile_index = 0
    
    print("Predicting Tiles...")
    for batch_number in progressbar.progressbar(range(0,num_batches)):
        img_batch,filenames=next(data_generator)
        pred_all= model.predict(img_batch)
        
        #pred = model.predict_generator(data_generator,steps=64)
        #reassemble(image_path,mask_path,pred,classes)
        for i in range(0,np.shape(pred_all)[0]):
            if(tile_index == len(all_tile_paths)):
                break
            predicted_mask = utils.onehot_to_rgb(pred_all[i],classes)
            img = img_batch[i] *255
            mask_pred_path = os.path.join(pred_tiles_dir,os.path.basename(filenames[i]))
            img_path = os.path.join(pred_tiles_dir,os.path.basename(filenames[i].replace(".png","a.png")))

            save_array_as_image(mask_pred_path, predicted_mask)
            save_array_as_image(img_path, img)
            tile_index += 1
            #mask = onehot_to_rgb(batch_mask[i],classes)
            #img = batch_img[i] *255
            
            #img_path = os.path.join("C:/Users/johan/Desktop/test", str(j) + "_" + str(i) + "_img.png")
            #mask_path = os.path.join("C:/Users/johan/Desktop/test", str(j) + "_" + str(i) + "_mask.png")

            #save_array_as_image(mask_path,mask)
            #save_array_as_image(img_path, img)


    
    print("Reassembling tiles Tiles...")
    for image_path in progressbar.progressbar(images_to_predict):
        dst_image_path = os.path.join(output_folder, os.path.basename(image_path))
        reassemble(image_path,dst_image_path,pred_tiles_dir)
        
    print("Cleaning up...")
    utils.delete_folder_contents(temp_dir)



    









        
if __name__ == '__main__':
    run("C:/Users/johan/Desktop/src_folder2/images","C:/Users/johan/Desktop/test")