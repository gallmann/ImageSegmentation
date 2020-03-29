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
    
