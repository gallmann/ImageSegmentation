# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:35:45 2020

@author: Johannes Gallmann
"""

import utils
import os
import geopandas as gpd
import gdal
import osr
from shapely.geometry import Polygon, MultiPolygon, LinearRing
from PIL import Image, ImageDraw
import numpy as np
import shutil
import constants
import progressbar

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








def save_array_as_image(image_path,image_array, tile_size = None):
    
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

        if not tile_size:
            gdal.Translate(image_path,ds, options=gdal.TranslateOptions(bandList=[1,2,3], format="png"))

        else:
            for i in range(0, width, tile_size):
                for j in range(0, height, tile_size):
                    #define paths of image tile and the corresponding json file containing the geo information
                    out_path_image = image_path[:-4] + "row" + str(int(j/tile_size)) + "_col" + str(int(i/tile_size)) + ".png"
                    #tile image with gdal (copy bands 1, 2 and 3)
                    gdal.Translate(out_path_image,ds, options=gdal.TranslateOptions(srcWin=[i,j,tile_size,tile_size], bandList=[1,2,3]))





def make_mask_image(image_path, mask_image_path, all_polygons):
    
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
    
    
    save_array_as_image_with_geo_coords(mask_image_path,image_path,mask)
    #save_array_as_image(mask_image_path, mask)


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



def tile_image(image_path, output_folder,src_dir_index, tile_size=256, overlap=0):
    
    """Tiles the image and the annotations into square shaped tiles of size tile_size
        Requires the image to have either a tablet annotation file (imagename_annotations.json)
        or the LabelMe annotation file (imagename.json) stored in the same folder

    Parameters:
        image_path (str): The image path 
        output_folder (string): Path of the output directory
        tile_size (int): the tile size 
        overlap (int): overlap in pixels to use during the image tiling process
    
    Returns:
        Fills the output_folder with all tiles (and annotation files in xml format)
        that contain any flowers.
    """
    
    #image = Image.open(image_path)
    image_array = utils.get_image_array(image_path)
    height = image_array.shape[0]
    width = image_array.shape[1]
    image_name = os.path.basename(image_path).replace("_mask.tif","").replace(".tif","")
    
    image_array = utils.get_image_array(image_path)
    height = image_array.shape[0]
    width = image_array.shape[1]

    
            
        
    currentx = 0
    currenty = 0
    while currenty < height:
        while currentx < width:       
            
            cropped_array = image_array[currenty:currenty+tile_size,currentx:currentx+tile_size,:3]
            
            if ("_mask.tif" in os.path.basename(image_path)):
                result = np.full((tile_size,tile_size,3),utils.name2color(classes,"Nothing"),dtype=np.uint8)
            else:
                result = np.full((tile_size,tile_size,3),0,dtype=np.uint8)
            result[:cropped_array.shape[0],:cropped_array.shape[1]] = cropped_array
                    
              
            #print(result.dtype)
            tile = Image.fromarray(result)

            #tile = image.crop((currentx,currenty,currentx + tile_size,currenty + tile_size))
            output_image_path = os.path.join(output_folder,  image_name + "_src_dir" + str(src_dir_index)  + "_subtile_" + "x" + str(currentx) + "y" + str(currenty) + "_size" + str(tile_size) + ".png")
            tile.save(output_image_path,"PNG")
                        
            currentx += tile_size-overlap
        currenty += tile_size-overlap
        currentx = 0
    
    

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
    
    '''
    val_mask_tiles_dir = os.path.join(training_data_dir,"val_masks")
    os.makedirs(val_mask_tiles_dir,exist_ok=True)

    val_image_tiles_dir = os.path.join(training_data_dir,"val_images")
    os.makedirs(val_image_tiles_dir,exist_ok=True)
    '''
    
    return (temp_dir,mask_tiles_dir,image_tiles_dir)


def resize_image_and_change_coordinate_system(image_path, dst_image_path, dst_gsd=constants.ground_sampling_distance):
    
    
    gdal_image = gdal.Open(image_path)
    width = gdal_image.RasterXSize
    #height = gdal_image.RasterYSize

    #Change Coordinate System of Image if necessary      
    geo_coordinates = utils.get_geo_coordinates(image_path)
    ground_sampling_size_x = (geo_coordinates.lr_lon - geo_coordinates.ul_lon) / width
    #ground_sampling_size_y = (geo_coordinates.ul_lat - geo_coordinates.lr_lat) / height
    
    dst_width = width * ground_sampling_size_x / dst_gsd
    
    proj = osr.SpatialReference(wkt=gdal_image.GetProjection())
    epsg_code_of_image = proj.GetAttrValue('AUTHORITY',1)
    
    
    if  abs(dst_width/width-1)>0.05:
        #projected_image_path = os.path.join(temp_dir,os.path.basename(image_path))
        gdal.Warp(dst_image_path,image_path,dstSRS='EPSG:'+str(EPSG_TO_WORK_WITH), width=dst_width)
    
    elif epsg_code_of_image != EPSG_TO_WORK_WITH:
        gdal.Warp(dst_image_path,image_path,dstSRS='EPSG:'+str(EPSG_TO_WORK_WITH))
    else:
        shutil.copyfile(image_path,dst_image_path)


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
        
        print("Tiling all images in input folder: " + src_dir)
        for image_path in progressbar.progressbar(utils.get_all_image_paths_in_folder(images_folder)):
            
            projected_image_path = os.path.join(temp_dir,os.path.basename(image_path))
            resize_image_and_change_coordinate_system(image_path,projected_image_path)
            image_path = projected_image_path
            '''
            #Change Coordinate System of Image if necessary      
            proj = osr.SpatialReference(wkt=gdal.Open(image_path).GetProjection())
            epsg_code_of_image = proj.GetAttrValue('AUTHORITY',1)
            if epsg_code_of_image != EPSG_TO_WORK_WITH:
                projected_image_path = os.path.join(temp_dir,os.path.basename(image_path))
                gdal.Warp(projected_image_path,image_path,dstSRS='EPSG:'+str(EPSG_TO_WORK_WITH))
                image_path = projected_image_path
            '''
            mask_image_path = os.path.join(temp_dir,os.path.basename(image_path).replace(".tif","_mask.tif"))
            
            all_polygons = get_all_polygons_from_shapefile(shape_file_path)
            all_polygons = convert_polygon_coords_to_pixel_coords(all_polygons,image_path)        
            make_mask_image(image_path,mask_image_path,all_polygons)
            tile_image(image_path,image_tiles_dir,src_dir_index)
            tile_image(mask_image_path,mask_tiles_dir,src_dir_index)

        
        #split_train_dir(image_tiles_dir, mask_tiles_dir, val_image_tiles_dir, val_mask_tiles_dir)
        
        utils.delete_folder_contents(temp_dir)
    shutil.rmtree(temp_dir)
    
    

run()










