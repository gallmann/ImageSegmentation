# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:35:45 2020

@author: Johannes Gallmann
"""

import utils
import os
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, LinearRing
from PIL import Image, ImageDraw
import numpy as np
import shutil
import constants
import progressbar
import split_dataset

EPSG_TO_WORK_WITH = constants.EPSG_TO_WORK_WITH

classes = ["Background", "Nothing"]


def get_all_polygons_from_shapefile(project_dir):
    
    gdf = gpd.read_file(project_dir)
    gdf = gdf[gdf.geometry.notnull()]
    gdf = gdf.to_crs({'init': 'EPSG:'+str(EPSG_TO_WORK_WITH)}) 
    
    all_polygons = []
    
    
    for index, row in gdf.iterrows():
        
        
        
        if type(row["geometry"]) is Polygon:
            
            
            '''
            #print(list(row["geometry"].exterior.coords))
            try:
                print(list(row["geometry"].interiors[0].coords))
            except:
                print("Exc")
            '''
            
            #TODO: interior
            all_polygons.append({"class_label": row["NUTZUNG"], "polygon": list(row["geometry"].exterior.coords), "interior_polygons":[]})
            
        elif type(row["geometry"]) is LinearRing:
            all_polygons.append({"class_label": row["NUTZUNG"], "polygon": list(row["geometry"].coords), "interior_polygons":[]})
        elif type(row["geometry"]) is MultiPolygon:
            for polygon in row["geometry"]:
                #TODO: interior
                all_polygons.append({"class_label": row["NUTZUNG"], "polygon": list(polygon.exterior.coords), "interior_polygons":[]})

        else:
            print("Unknown geometry type in input shape file ignored...")
    
    return all_polygons






def make_mask_image(image_path, mask_image_path, all_polygons, save_with_geo_coordinates = False):
    
    outer_polygons = []
    for polygon in all_polygons:
        outer_polygons.append(polygon["polygon"])
    
    
    # read image as RGB(A)
    img_array = utils.get_image_array(image_path)
    # create new image ("1-bit pixels, black and white", (width, height), "default color")
    mask_img = Image.new("RGB", (img_array.shape[1], img_array.shape[0]), utils.name2color(classes,"Background"))
    
    for polygon in all_polygons:
        color = utils.name2color(classes,polygon["class_label"])
        ImageDraw.Draw(mask_img).polygon(polygon["polygon"], outline=color, fill=color)
        #TODO: Inner polygon

    mask = np.array(mask_img)
    
    if (img_array.shape[2] == 4):
        alpha_mask = img_array[:,:,3] / 255
        
        # filtering image by mask
        mask[:,:,0] = mask[:,:,0] * alpha_mask + utils.name2color(classes,"Nothing")[0] * (1-alpha_mask)
        mask[:,:,1] = mask[:,:,1] * alpha_mask + utils.name2color(classes,"Nothing")[1] * (1-alpha_mask)
        mask[:,:,2] = mask[:,:,2] * alpha_mask + utils.name2color(classes,"Nothing")[2] * (1-alpha_mask)
    
    
    if save_with_geo_coordinates:
        utils.save_array_as_image_with_geo_coords(mask_image_path,image_path,mask)
    else:
        utils.save_array_as_image(mask_image_path, mask)





def convert_coordinates_to_pixel_coordinates(coords, image_width, image_height, target_geo_coords):
    
    geo_x = coords[0]
    geo_y = coords[1]
    rel_x_target = (geo_x-target_geo_coords.ul_lon)/(target_geo_coords.lr_lon-target_geo_coords.ul_lon)
    rel_y_target = 1-(geo_y-target_geo_coords.lr_lat)/(target_geo_coords.ul_lat-target_geo_coords.lr_lat)
    x_target = rel_x_target* image_width
    y_target = rel_y_target* image_height 
    return (x_target,y_target)


def convert_polygon_coords_to_pixel_coords(all_polygons, image_path):

    result_polygons = []
    
    target_geo_coords = utils.get_geo_coordinates(image_path,EPSG_TO_WORK_WITH)
    image = Image.open(image_path)
    width = image.size[0]
    height = image.size[1]
    
    for polygon in all_polygons:
        for index,coords in enumerate(polygon["polygon"]):
            pixel_coords = convert_coordinates_to_pixel_coordinates(coords,width,height,target_geo_coords)
            polygon["polygon"][index] = pixel_coords
        #TODO: Inner polygons
        result_polygons.append(polygon)

    return result_polygons


    

def add_shapefile_classes_to_label_dictionary(shape_file_path):
    all_polygons = get_all_polygons_from_shapefile(shape_file_path)
    
    for polygon in all_polygons:
        label = polygon["class_label"]
        if not label in classes:
            classes.append(label)
    classes.sort()

def make_folders(project_dir):
    
    temp_dir = os.path.join(project_dir,"temp")
    os.makedirs(temp_dir,exist_ok=True)
    utils.delete_folder_contents(temp_dir)
    
    training_data_dir = os.path.join(project_dir,"training_data")
    os.makedirs(training_data_dir,exist_ok=True)
    utils.delete_folder_contents(training_data_dir)

    mask_tiles_dir = os.path.join(training_data_dir,"masks")
    os.makedirs(mask_tiles_dir,exist_ok=True)

    image_tiles_dir = os.path.join(training_data_dir,"images")
    os.makedirs(image_tiles_dir,exist_ok=True)
        
    return (temp_dir,mask_tiles_dir,image_tiles_dir)



def run(src_dirs=constants.data_source_folders, working_dir=constants.working_dir):

    for src_dir_index,src_dir in enumerate(src_dirs):
        shape_file_path = os.path.join(src_dir,"shapes/shapes.shp")
        add_shapefile_classes_to_label_dictionary(shape_file_path)
    print(str(len(classes)) + " classes present in dataset:")
    print(classes)
    
    utils.save_obj(classes,os.path.join(working_dir,"labelmap.pkl"))
    
    
    (temp_dir,mask_tiles_dir,image_tiles_dir) = make_folders(working_dir)

    
    for src_dir_index,src_dir in enumerate(src_dirs):
    
        shape_file_path = os.path.join(src_dir,"shapes/shapes.shp")
        
        images_folder = os.path.join(src_dir,"images")
        
        print("Generating masks for all images in input folder: " + src_dir)
        
        

        for image_path in progressbar.progressbar(utils.get_all_image_paths_in_folder(images_folder)):
            
            
            projected_image_path = os.path.join(temp_dir,os.path.basename(image_path).replace(".tif","_srcdir" + str(src_dir_index) + ".tif"))
            utils.resize_image_and_change_coordinate_system(image_path,projected_image_path)
            image_path = projected_image_path
            
            mask_image_path = os.path.join(temp_dir,os.path.basename(image_path).replace(".tif","_mask.tif"))
            all_polygons = get_all_polygons_from_shapefile(shape_file_path)
            all_polygons = convert_polygon_coords_to_pixel_coords(all_polygons,image_path)   
                
            make_mask_image(image_path,mask_image_path,all_polygons)
            
            utils.tile_image(mask_image_path,mask_tiles_dir,classes, src_dir_index=src_dir_index, is_mask= True)
            utils.tile_image(image_path,image_tiles_dir,classes, src_dir_index=src_dir_index)

            


            
        utils.delete_folder_contents(temp_dir)
    shutil.rmtree(temp_dir)
    print("Splitting all training tiles and masks into train, validation and test sets...")
    split_dataset.split_into_train_val_and_test_sets(working_dir)
    
if __name__ == '__main__':
    run()










