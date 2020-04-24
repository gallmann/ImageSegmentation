# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 13:22:42 2020

@author: johan
"""


import constants
import utils
import os
import numpy as np
import progressbar
import gdal
import unet_utils

from PIL import Image, ImageDraw
import tensorflow as tf
tf.enable_eager_execution()



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


def create_color_legend(classes,save_path):
    
    class_height = 50
    
    
    image = Image.new("RGB", (600,len(classes) * class_height), (255,255,255))

    for i,clazz in enumerate(classes):
        
        color = utils.name2color(classes,clazz)
        color_polygon = [(10,i*class_height + 10),(40,i*class_height+10),(40,i*class_height+40),(10,i*class_height+40)]
        ImageDraw.Draw(image).polygon(color_polygon, outline=color, fill=color)
        ImageDraw.Draw(image).text((50,i*class_height+25), clazz + " " + str(color), fill=color)

    image.save(save_path)


def run(predict_folder,output_folder, working_dir=constants.working_dir, batch_size = constants.batch_size):

    
    [temp_dir, tiles_dir, pred_tiles_dir] = make_folders(working_dir)
    classes = utils.load_obj(constants.label_map)
    
    create_color_legend(classes, os.path.join(output_folder,"color_legend.png"))

    print("Preparing image tiles...",flush=True)
    images_to_predict = utils.get_all_image_paths_in_folder(predict_folder)
    for image_path in progressbar.progressbar(images_to_predict):
        utils.tile_image(image_path, tiles_dir, classes, tile_size=256, overlap=0)
    
    print("Loading Trained Model...",flush=True)
    
    all_tile_paths = utils.get_all_image_paths_in_folder(tiles_dir)

    #test_frames_dir = "C:/Users/johan/Desktop/working_dir/training_data/train_frames/0"
    #test_masks_dir = "C:/Users/johan/Desktop/working_dir/training_data/train_masks/0"
    
    
    
    #print(len(classes))
    
    model = unet_utils.get_small_unet(n_filters = 32,num_classes=len(classes),batch_size=batch_size)
    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=[unet_utils.tversky_loss,unet_utils.dice_coef,'accuracy'])
    #model.summary()
    
    model_save_path = constants.trained_model
    
    model.load_weights(model_save_path)
    data_generator = unet_utils.DataGeneratorWithFileNames(tiles_dir,classes,batch_size=batch_size)
    
    num_batches = int(np.ceil(float(len(all_tile_paths)) / float(batch_size)))
    tile_index = 0
    
    print("Predicting Tiles...",flush=True)
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


    
    print("Reassembling Tiles...",flush=True)
    for image_path in progressbar.progressbar(images_to_predict):
        dst_image_path = os.path.join(output_folder, os.path.basename(image_path))
        reassemble(image_path,dst_image_path,pred_tiles_dir)
        
    print("Cleaning up...",flush=True)
    utils.delete_folder_contents(temp_dir)



    









        
if __name__ == '__main__':
    run(constants.folder_with_images_to_predict,constants.predictions_output_folder)