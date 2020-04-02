# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:40:58 2020

@author: johan
"""

import os
import shutil
import gdal
import numpy as np
import osr
import pyproj
import pickle
import constants
from PIL import Image


class GeoInformation(object):
    def __init__(self,dictionary=None):
        if not dictionary:
            self.lr_lon = 0
            self.lr_lat = 0
            self.ul_lon = 0
            self.ul_lat = 0
        else:
            for key in dictionary:
                setattr(self, key, dictionary[key])


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
        encoded_image[:,:,i] = np.all(rgb_image.reshape( (-1,3) ) == id2color(classes,i), axis=1).reshape(shape[:2])
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
        output[single_layer==name2id(classes,c)] = name2color(classes,c)
    return np.uint8(output)



def save_obj(obj, path ):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)




def get_all_image_paths_in_folder(folder_path):
    """Finds all images (png, jpg or tif) inside a folder

    Parameters:
        folder_path (str): the folder path to look for images inside
    
    Returns:
        list: a list of image_paths (strings)
    """
    
    images = []
    if not os.path.isdir(folder_path):
        return images
    for file in os.listdir(folder_path):
        if file.lower().endswith(".png") or file.lower().endswith(".jpg") or file.lower().endswith(".tif"):
            images.append(os.path.join(folder_path, file))
    return images

colors = [(64, 128, 64),
          (192, 0, 128),
          (0, 128, 192),
          (0, 128, 64),
          (128, 0, 0),
          (64, 0, 128),
          (64, 0, 192),
          (192, 128, 64),
          (192, 192, 128),
          (64, 64, 128),
          (128, 0, 192),
          (192, 0, 64),
          (128, 128, 64),
          (192, 0, 192),
          (128, 64, 64),
          (64, 192, 128),
          (64, 64, 0),
          (128, 64, 128),
          (128, 128, 192),
          (0, 0, 192),
          (192, 128, 128),
          (128, 128, 128),
          (64, 128, 192),
          (0, 0, 64),
          (0, 64, 64),
          (192, 64, 128),
          (128, 128, 0),
          (192, 128, 192),
          (64, 0, 64),
          (192, 192, 0),
          (0, 0, 0),
          (64, 192, 0)]


def name2id(classes, name):
    return classes.index(name)

def id2name(classes,idx):
    return classes[idx]

def id2color(classes,idx):
    if idx >= len(colors):
        print("WARNING: TOO FEW COLORS DEFINED TO HANDLE SO MANY CLASSES")
    return colors[idx%len(colors)]

def color2id(classes,color):
    return colors.index(color)    

def name2color(classes,name):
    return id2color(classes,name2id(classes,name))

def color2name(classes,color):
    return id2name(classes,color2id(classes,color))

def delete_folder_contents(folder_path):
    """Deletes all files and subfolders within a folder

    Parameters:
        folder_path (str): the folder path to delete all contents in
    
    Returns:
        None
    """

    for the_file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

    
    
def get_image_array(image_path, xoff=0, yoff=0, xsize=None, ysize=None):
    '''
    try:
        image = Image.open(image_path)
        width, height = image.size
        img = Image.open(image_path).convert("RGB")
        img_array = numpy.asarray(img)
        return img_array
    except Image.DecompressionBombError:
    '''
    ds = gdal.Open(image_path)
    image_array = ds.ReadAsArray(xoff, yoff, xsize, ysize).astype(np.uint8)
    image_array = np.swapaxes(image_array,0,1)
    image_array = np.swapaxes(image_array,1,2)
    return image_array



def get_geo_coordinates(input_image, epsg_code=2056):
    """
    Reads the geo coordinates of the upper-left and lower-right corner of the image and coverts them
    to the lv95+ format (if not already) and returns it as a GeoInformation object. The input image must
    be in the jpg or png format with a imagename_geoinfo.json file in the same folder
    or otherwise can be a georeferenced tif.

    
    Parameters:
        input_image (str): path to the image
        
    Returns:
        GeoInformation: GeoInformation object containing the upper-left and lower-right geo coordinates
            in lv95+ coordinate system
    """

    try:
        #if the input_image is a geo-annotated .tif file, read the geo information using gdal
        ds = gdal.Open(input_image)
        inSRS_wkt = ds.GetProjection()  # gives SRS in WKT
        inSRS_converter = osr.SpatialReference()  # makes an empty spatial ref object
        inSRS_converter.ImportFromWkt(inSRS_wkt)  # populates the spatial ref object with our WKT SRS
        inSRS_forPyProj = inSRS_converter.ExportToProj4()  # Exports an SRS ref as a Proj4 string usable by PyProj
        
        input_coord_system = pyproj.Proj(inSRS_forPyProj) 
        swiss = pyproj.Proj("+init=EPSG:" + str(epsg_code))
        
        ulx, xres, xskew, uly, yskew, yres  = ds.GetGeoTransform()
        lrx = ulx + (ds.RasterXSize * xres)
        lry = uly + (ds.RasterYSize * yres)
        geo_info = GeoInformation()
        geo_info.lr_lon = lrx
        geo_info.lr_lat = lry
        geo_info.ul_lon = ulx
        geo_info.ul_lat = uly
        geo_info.lr_lon,geo_info.lr_lat = pyproj.transform(input_coord_system, swiss, geo_info.lr_lon, geo_info.lr_lat)
        geo_info.ul_lon,geo_info.ul_lat = pyproj.transform(input_coord_system, swiss, geo_info.ul_lon, geo_info.ul_lat)

        return geo_info
    except RuntimeError:
        return None
    


def resize_image_and_change_coordinate_system(image_path, dst_image_path, dst_gsd=constants.ground_sampling_distance, epsg_to_work_with=constants.EPSG_TO_WORK_WITH):
    
    
    gdal_image = gdal.Open(image_path)
    width = gdal_image.RasterXSize
    #height = gdal_image.RasterYSize

    #Change Coordinate System of Image if necessary      
    geo_coordinates = get_geo_coordinates(image_path)
    ground_sampling_size_x = (geo_coordinates.lr_lon - geo_coordinates.ul_lon) / width
    #ground_sampling_size_y = (geo_coordinates.ul_lat - geo_coordinates.lr_lat) / height
    
    dst_width = width * ground_sampling_size_x / dst_gsd
    
    proj = osr.SpatialReference(wkt=gdal_image.GetProjection())
    epsg_code_of_image = proj.GetAttrValue('AUTHORITY',1)
    
    
    if  abs(dst_width/width-1)>0.05:
        #projected_image_path = os.path.join(temp_dir,os.path.basename(image_path))
        gdal.Warp(dst_image_path,image_path,dstSRS='EPSG:'+str(epsg_to_work_with), width=dst_width)
    
    elif epsg_code_of_image != epsg_to_work_with:
        gdal.Warp(dst_image_path,image_path,dstSRS='EPSG:'+str(epsg_to_work_with))
    else:
        shutil.copyfile(image_path,dst_image_path)


def tile_image(image_path, output_folder, classes, src_dir_index=0, is_mask=False, tile_size=256, overlap=0):
        
    image_array = get_image_array(image_path)
    height = image_array.shape[0]
    width = image_array.shape[1]
    image_name = os.path.basename(image_path).replace("_mask.tif","").replace(".tif","")
        
            
        
    currentx = 0
    currenty = 0
    while currenty < height:
        while currentx < width:       
            
            cropped_array = image_array[currenty:currenty+tile_size,currentx:currentx+tile_size,:3]
            
            if is_mask:
                result = np.full((tile_size,tile_size,3),name2color(classes,"Nothing"),dtype=np.uint8)
            else:
                result = np.full((tile_size,tile_size,3),0,dtype=np.uint8)
            result[:cropped_array.shape[0],:cropped_array.shape[1]] = cropped_array
                    
              
            tile = Image.fromarray(result)

            output_image_path = os.path.join(output_folder,  image_name + "_subtile_" + "x" + str(currentx) + "_y" + str(currenty) + ".png")
            tile.save(output_image_path,"PNG")
                        
            currentx += tile_size-overlap
        currenty += tile_size-overlap
        currentx = 0


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


def save_array_as_image(image_path,image_array):
    
    image_array = image_array.astype(np.uint8)
    if not image_path.endswith(".png") and not image_path.endswith(".jpg") and not image_path.endswith(".tif"):
        print("Error! image_path has to end with .png, .jpg or .tif")
    height = image_array.shape[0]
    width = image_array.shape[1]
    if height*width < Image.MAX_IMAGE_PIXELS:
        newIm = Image.fromarray(image_array, "RGB")
        newIm.save(image_path)
    
    else:
        gdal.AllRegister()
        driver = gdal.GetDriverByName( 'MEM' )
        ds1 = driver.Create( '', width, height, 3, gdal.GDT_Byte)
        ds = driver.CreateCopy(image_path, ds1, 0)
            
        image_array = np.swapaxes(image_array,2,1)
        image_array = np.swapaxes(image_array,1,0)
        ds.GetRasterBand(1).WriteArray(image_array[0], 0, 0)
        ds.GetRasterBand(2).WriteArray(image_array[1], 0, 0)
        ds.GetRasterBand(3).WriteArray(image_array[2], 0, 0)
        gdal.Translate(image_path,ds, options=gdal.TranslateOptions(bandList=[1,2,3], format="png"))
