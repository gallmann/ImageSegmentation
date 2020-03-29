# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:21:15 2020

@author: johan
"""

from PIL import Image
import utils
import os
import progressbar
import gdal


def resize_pil(image_path, dest_path, new_width, new_height):
    im = Image.open(image_path) 
    im_resized = im.resize((new_width,new_height), Image.ANTIALIAS)
    im_resized.save(dest_path)


def resize_gdal(image_path, dest_path, new_width, new_height):
    ds = gdal.Open(image_path)
    gdal.Translate(dest_path,ds, options=gdal.TranslateOptions(width=int(new_width),height=int(new_height)))


for image_path in progressbar.progressbar(utils.get_all_image_paths_in_folder("C:/Users/johan/Desktop/src_folder2/images")):
    dest_path = "C:/Users/johan/Desktop/src_folder2/images_low_res/" + os.path.basename(image_path)
    resize_gdal(image_path,dest_path,128,128)