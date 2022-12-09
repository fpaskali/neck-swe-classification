#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 1 09:05:10 2022

@author: fpaskali
"""
import csv, time, PIL, argparse
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import io, color
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, erosion
from skimage.color import rgb2gray
from skimage.filters import gabor_kernel
from skimage.util import img_as_float

class Image(object):
    
    def __init__(self, image_path, qmap_path, subject, repeat, group, task):
        """
        Store image data, used to extract features from an image.

        Parameters
        ----------
        image_path : str
            path of the image.
        qmap_path : str
            path of the quality map.
        subject : int
            Subject id.
        repeat : int
            the number of session.
        group : int
            0 = pain, 1 = control.
        task : str
            task description.
            
        Attributes
        ----------
        color_perc : float
            the percentage of colored pixels in the image.
        features : dict
            dictionary containing all features that are extracted from the image.
        miny : int
            the row number where region of interest starts.  
        minx : int
            the column number where region of interest starts.
        maxy : int
            the row number where region of interest ends.
        maxx : int
            the column number where region of interest ends.
        image_array : numpy array
            numpy array of the image.
        qmap_array : numpy array
            numpy array of the quality map.
        """
        self.image_path = image_path
        self.qmap_path = qmap_path
        self.subject = subject
        self.repeat = repeat
        self.group = group
        self.task = task
        
        self.valid_pix = None
        self.color_perc = 0
        self.features = {}
        
        self.miny = 0
        self.minx = 0
        self.maxy = 0
        self.maxx = 0
        
        self.image_array = None
        self.qmap_array = None
        
        self._read_initial_features()
        
    def load_image(self):
        """
        Load the image from image path parameter.
        """
        self.image_array = io.imread(self.image_path)
    
    def load_qmap(self):
        """
        Load the quality map from qmap path parameter.
        """
        self.qmap_array = io.imread(self.qmap_path)

    def crop_roi(self, bottom_only=False, leave_out_last_segment=False):
        """
        Crop the region of interest in elastography image.

        Parameters
        ----------
        bottom_only : bool
            if true, crop only the bottom half.
        """
        height = (self.maxy - self.miny)
        
        if leave_out_last_segment:
            cut = height // 10
        else:
            cut = 0
        
        if bottom_only:
            half_h = height // 2
            self.image_array = self.image_array[self.miny+half_h:self.maxy-cut,
                                                self.minx:self.maxx]
        else:
            self.image_array = self.image_array[self.miny:self.maxy-cut,
                                                self.minx:self.maxx]
    
    def _read_initial_features(self):
        """
        Write initial info into features dictionary.
        """
        self.features['name'] = self.image_path
        self.features['subject'] = self.subject
        self.features['group'] = self.group
        self.features['repeat'] = self.repeat
        self.features['task'] = self.task
    
    def detect_colored_region(self):
        """
        Read the image and convert it to HSV image. Otsu threshold is applied
        on the saturation layer and enchance the threshold to remove artifacts
        connected to image borders. Then label the region of interest and save
        the parameters to miny, minx, maxy, maxx.
        """
        hsv_image = color.rgb2hsv(self.qmap_array)
        sat_image = hsv_image[:,:,1]

        thresh = 0.4
        bw = closing(sat_image > thresh, square(2))
        selem = square(2)
        eroded = erosion(bw, selem)
        
        label_image = label(eroded)

        for region in regionprops(label_image):
            if region.area >= 6000:
                self.miny, self.minx, self.maxy, self.maxx = region.bbox

    def add_feature_value(self, feature_name, value):
        """
        Helper function to add feature to self.features dictionary.

        Parameters
        ----------
        feature_name : str
            the name of the feature.
        value : int or str
            quantitative or qualitative value.
        """        
        self.features[feature_name] = value

class FeatureExtractor(object):
    """
    Extract features for machine learning.

    Parameters
    ----------
    csv_file : str
        path to csv file containing images info.
    remove_low_color : bool
        if true remove images with color pixels less than 50%.
    """
    def __init__(self, csv_file, bottom_only=False, remove_low_color=False):
        self.csv_file = csv_file
        self.bottom_only = bottom_only
        self.remove_low_color = remove_low_color
        self._load_dataset_csv()
        self._clean_image_list()

        if bottom_only:
            self.seg_num = 5
            print("Using bottom part of ROI only.")
        else:
            self.seg_num = 10
            print("Using whole region of interest.")
    
    def _load_dataset_csv(self):
        """
        Check csv file structure. If it is correct, load the data for all images.
        """        
        lista = []
        dataset = pd.read_csv(self.csv_file, keep_default_na=False)
        
        if dataset.columns.size >= 5 and list(dataset.columns[:3]) == ['subject', 'repeat', 'group']:
            for i in range(3, dataset.columns.size, 2):
                for row in range(dataset.index.size):
                    lista.append(Image(dataset.iloc[row, i],
                                       dataset.iloc[row, i+1],
                                       dataset.iloc[row, 0],
                                       dataset.iloc[row, 1],
                                       dataset.iloc[row, 2],
                                       dataset.columns[i]))
            self.images = lista
        else:
            print("Invalid dataset file. The structure of the csv file is not correct")
   
    def _clean_image_list(self):
        """
        Remove Image objects without elastography images or without quality maps.
        """
        self.images = [x for x in self.images if x.image_path != '']
        self.images = [x for x in self.images if x.qmap_path != '']
    
    def _load_and_crop_images(self):
        """
        Remove Image objects without elastography images or without quality maps.
        """
        i = 0
        for img in self.images:
            img.load_image()
            img.crop_roi(self.bottom_only)
            i += 1
            print(f'\rLoading images...[{i}/{len(self.images)}]', end='')
        print()
        
    def _detect_roi(self):
        i = 0
        for img in self.images:
            img.load_qmap()
            img.detect_colored_region()
            del img.qmap_array
            i += 1
            print(f'\rROI calculation...[{i}/{len(self.images)}]', end='')
        print()
    
    def extract_images_feature(self, function, name):
        i = 0
        for img in self.images:
            img.add_feature_value(name, function(img.image_array))
            i += 1
            print(f"\r[Extracting ({name}): {i/len(self.images):.0%}]", end='')
        print()
        
    def extract_segments_feature(self, function, name, segment_num):
        i = 0 
        for img in self.images:
            seg_count = 0
            for ymin, ymax in self._h_segment_image(img.image_array, segment_num):
                seg_count += 1
                img.add_feature_value(f'{name}_hsegment_{seg_count}/{segment_num}',
                                      function(img.image_array[ymin:ymax,:]))
                i += 1
                print(f"\r[Extracting ({name}_segments): {i/(len(self.images)*self.seg_num):.0%}]", end='')
        print()
        
    def _h_segment_image(self, image, num):
        segments = []
        height = image.shape[0]
        y = height // num
        for i in range(num):
            segments.append((y*i, y*(i+1)))
        return segments
        
    def _generate_red_color_mask(self, image):
        """
        Generate a binary label image of red regions and store it in 
        self.red_image_array attribute.
        """
        image = PIL.Image.fromarray(image)
        hsv = image.convert('HSV')    
        
        x, y = hsv.size
        red_image = np.zeros((y, x))
        for i in range(x):
            for j in range(y):
                h, s, v = hsv.getpixel((i,j))
                if (h <= 68 or h >= 360) and s >= 55 and v >= 55:
                    red_image[j,i] = 1
                    
        return red_image

    ## Methods for feature extraction
       
    def image_mean(self, image):
        return np.nanmean(image.flatten())

    def image_median(self, image):
        return np.nanmedian(image.flatten())

    def image_sd(self, image):
        return np.nanstd(image.flatten())

    def color_ry(self, image):
        image = PIL.Image.fromarray(image)
        hsv = image.convert('HSV')    
        
        x, y = hsv.size
        red_and_yellow_image = np.zeros((y, x))
        for i in range(x):
            for j in range(y):
                h, s, v = hsv.getpixel((i,j))
                if (h <= 68 or h >= 360) and s >= 55 and v >= 55:
                    red_and_yellow_image[j,i] = 1
        return red_and_yellow_image.sum()/red_and_yellow_image.size

    def color_g(self, image):
        image = PIL.Image.fromarray(image)
        hsv = image.convert('HSV')    
        
        x, y = hsv.size
        green_image = np.zeros((y, x))
        for i in range(x):
            for j in range(y):
                h, s, v = hsv.getpixel((i,j))
                if (h >= 69 and h <= 168) and s >= 55 and v >= 55:
                    green_image[j,i] = 1
        return green_image.sum()/green_image.size

    def color_b(self, image):
        image = PIL.Image.fromarray(image)
        hsv = image.convert('HSV')    
        
        x, y = hsv.size
        blue_image = np.zeros((y, x))
        for i in range(x):
            for j in range(y):
                h, s, v = hsv.getpixel((i,j))
                if (h >= 169 and h <= 240) and s >= 55 and v >= 55:
                    blue_image[j,i] = 1
        return blue_image.sum()/blue_image.size

    def red_layer_value_sum(self, image):
        image = image[:,:,0]
        return int(np.nansum(image.flatten()))

    def red_layer_value_mean(self, image):
        image = image[:,:,0]
        return np.nanmean(image.flatten())
    
    def red_layer_value_median(self, image):
        image = image[:,:,0]
        return np.nanmedian(image.flatten())
    
    def green_layer_value_sum(self, image):
        image = image[:,:,1]
        return int(np.nansum(image.flatten()))
    
    def green_layer_value_mean(self, image):
        image = image[:,:,1]
        return np.nanmean(image.flatten())
    
    def green_layer_value_median(self, image):
        image = image[:,:,1]
        return np.nanmedian(image.flatten())
    
    def blue_layer_value_sum(self, image):
        image = image[:,:,2]
        return int(np.nansum(image.flatten()))
    
    def blue_layer_value_mean(self, image):
        image = image[:,:,2]
        return np.nanmean(image.flatten())
    
    def blue_layer_value_median(self, image):
        image = image[:,:,2]
        return np.nanmedian(image.flatten())
    
    def grayscale_value(self, image):
        image = rgb2gray(image)
        return np.nansum(image.flatten())
    
    def grayscale_value_mean(self, image):
        image = rgb2gray(image)
        return np.nanmean(image.flatten())
    
    def grayscale_value_median(self, image):
        image = rgb2gray(image)
        return np.nanmedian(image.flatten())

    def grayscale_value_min(self, image):
        image = rgb2gray(image)
        return np.nanmin(image.flatten())

    def grayscale_value_max(self, image):
        image = rgb2gray(image)
        return np.nanmax(image.flatten())
    
    def gabor_kernel1_mean(self, image):
        image = img_as_float(rgb2gray(image))
        gk = np.real(gabor_kernel(frequency=0.1, theta=1/4.*np.pi))
        filtered = ndi.convolve(image, gk, mode='wrap')
        return np.nanmean(filtered)
        
    def gabor_kernel1_var(self, image):
        image = img_as_float(rgb2gray(image))
        gk = np.real(gabor_kernel(frequency=0.1, theta=1/4.*np.pi))
        filtered = ndi.convolve(image, gk, mode='wrap')
        return np.nanvar(filtered)
    
    def gabor_kernel2_mean(self, image):
        image = img_as_float(rgb2gray(image))
        gk = np.real(gabor_kernel(frequency=0.1, theta=2/4.*np.pi))
        filtered = ndi.convolve(image, gk, mode='wrap')
        return np.nanmean(filtered)
        
    def gabor_kernel2_var(self, image):
        image = img_as_float(rgb2gray(image))
        gk = np.real(gabor_kernel(frequency=0.1, theta=2/4.*np.pi))
        filtered = ndi.convolve(image, gk, mode='wrap')
        return np.nanvar(filtered)
    
    def gabor_kernel3_mean(self, image):
        image = img_as_float(rgb2gray(image))
        gk = np.real(gabor_kernel(frequency=0.1, theta=3/4.*np.pi))
        filtered = ndi.convolve(image, gk, mode='wrap')
        return np.nanmean(filtered)
        
    def gabor_kernel3_var(self, image):
        image = img_as_float(rgb2gray(image))
        gk = np.real(gabor_kernel(frequency=0.1, theta=3/4.*np.pi))
        filtered = ndi.convolve(image, gk, mode='wrap')
        return np.nanvar(filtered)
    
    def gabor_kernel4_mean(self, image):
        image = img_as_float(rgb2gray(image))
        gk = np.real(gabor_kernel(frequency=0.1, theta=np.pi))
        filtered = ndi.convolve(image, gk, mode='wrap')
        return np.nanmean(filtered)
        
    def gabor_kernel4_var(self, image):
        image = img_as_float(rgb2gray(image))
        gk = np.real(gabor_kernel(frequency=0.1, theta=np.pi))
        filtered = ndi.convolve(image, gk, mode='wrap')
        return np.nanvar(filtered)

    def measure_y_coor_relative(self, image):
        """
        Detect centroid of red regions and calculate the mean value of y coordinates.
        Then add it to features dictionary.
        """
        label_image = label(self._generate_red_color_mask(image))
        
        depths = []
        areas = []
        for region in regionprops(label_image):
            areas.append(region.area)
            depths.append(region.centroid[0])
        
        if depths:
            depths = np.array(depths)
            areas = np.array(areas)
            weights = areas / np.nansum(areas)
            
            weighted_mean = (depths * weights).sum()
            return weighted_mean/label_image.shape[0]
        else:
            return np.nan

    def measure_hs_region_area_relative(self, image):
        """
        Detect centroid of red regions and calculate the mean value of y coordinates.
        Then add it to features dictionary.
        """
        label_image = label(self._generate_red_color_mask(image))
        
        areas = []
        for region in regionprops(label_image):
            areas.append(region.area/image.size)
        
        if areas:
            return np.nanmean(areas)
        else:
            return np.nan
    
    ########################################
    
    def feature_extraction(self, whole_roi, segment):
        """
        Extract features and features dictionary.

        Parameters
        ----------
        whole_roi : bool
            extract features from region of interest.
        segment : bool
            segment region of interest and extract features of each segment.
        """
        if whole_roi:
            print("Extracting features of region of interest.")
            self.extract_images_feature(self.image_mean, 'mean')
            self.extract_images_feature(self.image_median, 'median')
            self.extract_images_feature(self.image_sd, 'sd')
            self.extract_images_feature(self.color_ry, 'red_pixels')
            self.extract_images_feature(self.color_g, 'green_pixels')
            self.extract_images_feature(self.color_b, 'blue_pixels')
            self.extract_images_feature(self.red_layer_value_sum, 'red_layer_value_sum')
            self.extract_images_feature(self.red_layer_value_mean, 'red_layer_value_mean')
            self.extract_images_feature(self.red_layer_value_median, 'red_layer_value_median')
            self.extract_images_feature(self.green_layer_value_sum, 'green_layer_value_sum')
            self.extract_images_feature(self.green_layer_value_mean, 'green_layer_value_mean')
            self.extract_images_feature(self.green_layer_value_median, 'green_layer_value_median')
            self.extract_images_feature(self.blue_layer_value_sum, 'blue_layer_value_sum')
            self.extract_images_feature(self.blue_layer_value_mean, 'blue_layer_value_mean')
            self.extract_images_feature(self.blue_layer_value_median, 'blue_layer_value_median')
            self.extract_images_feature(self.grayscale_value, 'gray_value')
            self.extract_images_feature(self.grayscale_value_mean, 'gray_value_mean')
            self.extract_images_feature(self.grayscale_value_median, 'gray_value_median')
            self.extract_images_feature(self.grayscale_value_min, 'gray_value_min')
            self.extract_images_feature(self.grayscale_value_max, 'gray_value_max')
            self.extract_images_feature(self.gabor_kernel1_mean, 'gabor_kernel1_mean')
            self.extract_images_feature(self.gabor_kernel1_var, 'gabor_kernel1_var')
            self.extract_images_feature(self.gabor_kernel2_mean, 'gabor_kernel2_mean')
            self.extract_images_feature(self.gabor_kernel2_var, 'gabor_kernel2_var')
            self.extract_images_feature(self.gabor_kernel3_mean, 'gabor_kernel3_mean')
            self.extract_images_feature(self.gabor_kernel3_var, 'gabor_kernel3_var')
            self.extract_images_feature(self.gabor_kernel4_mean, 'gabor_kernel4_mean')
            self.extract_images_feature(self.gabor_kernel4_var, 'gabor_kernel4_var')
            self.extract_images_feature(self.measure_y_coor_relative, "depth_rel")
            self.extract_images_feature(self.measure_hs_region_area_relative, "hs_area_rel")
        if segment:
            print("Extracting features of horizontal segments of region of interest.")
            self.extract_segments_feature(self.image_mean, 'mean', self.seg_num)
            self.extract_segments_feature(self.image_median, 'median', self.seg_num)
            self.extract_segments_feature(self.image_sd, 'sd', self.seg_num)
            self.extract_segments_feature(self.color_ry, 'red_pixels', self.seg_num)
            self.extract_segments_feature(self.color_g, 'green_pixels', self.seg_num)
            self.extract_segments_feature(self.color_b, 'blue_pixels', self.seg_num)
            self.extract_segments_feature(self.red_layer_value_sum, 'red_layer_value_sum',self.seg_num)
            self.extract_segments_feature(self.red_layer_value_mean, 'red_layer_value_mean',self.seg_num)
            self.extract_segments_feature(self.red_layer_value_median, 'red_layer_value_median',self.seg_num)
            self.extract_segments_feature(self.green_layer_value_sum, 'green_layer_value_sum', self.seg_num)
            self.extract_segments_feature(self.green_layer_value_mean, 'green_layer_value_mean', self.seg_num)
            self.extract_segments_feature(self.green_layer_value_median, 'green_layer_value_median', self.seg_num)
            self.extract_segments_feature(self.blue_layer_value_sum, 'blue_layer_value_sum', self.seg_num)
            self.extract_segments_feature(self.blue_layer_value_mean, 'blue_layer_value_mean', self.seg_num)
            self.extract_segments_feature(self.blue_layer_value_median, 'blue_layer_value_median', self.seg_num)
            self.extract_segments_feature(self.grayscale_value, 'gray_value', self.seg_num)
            self.extract_segments_feature(self.grayscale_value_mean, 'gray_value_mean',self.seg_num)
            self.extract_segments_feature(self.grayscale_value_median, 'gray_value_median',self.seg_num)
            self.extract_segments_feature(self.grayscale_value_min, 'gray_value_min', self.seg_num)
            self.extract_segments_feature(self.grayscale_value_max, 'gray_value_max', self.seg_num)
            self.extract_segments_feature(self.gabor_kernel1_mean, 'gabor_kernel1_mean', self.seg_num)
            self.extract_segments_feature(self.gabor_kernel1_var, 'gabor_kernel1_var', self.seg_num)
            self.extract_segments_feature(self.gabor_kernel2_mean, 'gabor_kernel2_mean', self.seg_num)
            self.extract_segments_feature(self.gabor_kernel2_var, 'gabor_kernel2_var', self.seg_num)
            self.extract_segments_feature(self.gabor_kernel3_mean, 'gabor_kernel3_mean', self.seg_num)
            self.extract_segments_feature(self.gabor_kernel3_var, 'gabor_kernel3_var', self.seg_num)
            self.extract_segments_feature(self.gabor_kernel4_mean, 'gabor_kernel4_mean', self.seg_num)
            self.extract_segments_feature(self.gabor_kernel4_var, 'gabor_kernel4_var', self.seg_num)
            self.extract_segments_feature(self.measure_y_coor_relative, "depth_rel", self.seg_num)
            self.extract_segments_feature(self.measure_hs_region_area_relative, "hs_area_rel", self.seg_num)

    def _check_color_perc(self):
        """
        Checking percentage of colored pixels in the images.
        """
        i = 0
        for img in self.images:
            img.check_color_perc()
            i += 1
            print(f'\rChecking color pixels perc...[{i}/{len(self.images)}]', end='')
        print()

    def preprocess(self):
        """
        Prepare the images for feature extraction.
        """
        self._detect_roi()
        self._load_and_crop_images()
        if self.remove_low_color:
            self._check_color_perc()
            self.images = [x for x in self.images if x.color_perc > 0.5]
        
    def write_csv(self, filename):
        """
        Save the extracted features in csv file.

        Parameters
        ----------
        filename : str
            the name of csv file to save the images.
        """
        with open(f'{filename}', mode = 'w') as csv_file:
            feature_names = list(set(self.images[0].features.keys()).difference(set(['name', 'subject', 'task', 'repeat', 'group'])))
            feature_names.sort()
            fieldnames = ['name', 'subject', 'repeat', 'task'] + feature_names + ['group']
            file_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)            
            file_writer.writeheader()
            count = 0
            for img in self.images:
                count += 1
                file_writer.writerow(img.features)
                print(f"\r[Please wait: {count/len(self.images):.0%}]", end='')
            print()
    
    def main(self, whole_roi, segment):
        self.preprocess()
        self.feature_extraction(whole_roi, segment)
        self.write_csv(f'{time.strftime("%Y%m%d_%H%M")}_features.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Feature extraction script.")
    parser.add_argument("-csv", help="path to dataset csv file.")
    parser.add_argument("-roi", help="Extract features of roi.", action='store_true')
    parser.add_argument("-seg", help="Segment roi and extract features.", action='store_true')
    parser.add_argument("-remove", help="Remove images with less than 50 percent colored pixels.", action='store_true', default=False)
    parser.add_argument('-bottom', help="Extract features from bottom part only.", action='store_true', default=False)

    args = parser.parse_args()
    
    extractor = FeatureExtractor(args.csv, args.bottom, args.remove)
    if not(args.roi or args.seg):
        print("No extraction mode chosen. The extraction cannot be performed. Choose -roi or -seg, or both.")
    else:
        extractor.main(args.roi, args.seg)

