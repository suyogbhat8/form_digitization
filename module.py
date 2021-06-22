#!/usr/bin/env python
# coding: utf-8

import cv2
import pandas as pd
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import sys
import skimage.io as skio
from skimage.transform import rescale, resize
from decimal import Decimal, ROUND_HALF_UP
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from typing import Tuple, Union
import math
from deskew import determine_skew
from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity
from skimage.io import imread, imsave
from imutils import contours
import PyPDF2
#to rotate the skewed images
def rotate(image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)


def preproc(inn,joint):
    #page number and their label types
    if inn=='1':
        nanu=0.126                                                                                   #relative cropping length below last dark strip
        lp=70                                                                                        #number of fields in the page
        sati=[0,1,2,4,16,25,33,34,35,36,37,50,66,59,67,68]                                           #numeric
        char=[3,11,12,13,14,15,17,23,24,45,46,47,48,49,63,64,65]                                     #alphabetic
        chek=[5,6,7,8,9,10,18,19,20,21,22,27,31,38,39,40,41,42,43,51,52,53,54,55,56,57,58]           #alpha numeric
        boxx=[0,1,2,3,4,11,12,13,14,15,16,17,23,24,25,26,28,29,30,32,33,34,35,36,
              37,44,45,46,47,48,49,50,59,60,61,62,63,64,65,66,67,68,69,70]
        printht=20
        vert=15
    elif inn=='2':
        nanu=1.05
        lp=56
        sati=[31,29,44,54,35,50]
        char=[3,41,45,46,47,48,49,51,52,55]
        chek=[0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,23,24,25,26,27,28,30,32,33,34,36,37,38,39,40,42,43,53]
        boxx=[3,21,29,31,35,41,44,45,46,47,48,49,50,51,52,54,55,56]
        printht=20
        vert=15
    elif inn=='3':
        nanu=4.15
        lp=5
        sati=[4]
        char=[2,3]
        chek=[0,1]
        boxx=[4]
        printht=15
        vert=15
    else:
        nanu=5.35
        lp=42
        sati = [0,3, 11, 7,8, 13, 14, 15, 17, 18, 19, 20, 21, 22,23, 25, 29, 30, 31, 32, 33, 40, 41]
        char = [2, 12,  9, 10, 16, 38, 39]
        chek = [1, 4, 5, 6, 24, 25, 26, 27, 28, 34, 35, 36, 37]
        boxx = [0, 7, 25, 29, 30, 31, 32, 33, 41]
        printht=17
        vert=13
    if joint and inn==1:
        nanu=0.126
        lp=70                                                                                        #number of fields in the page
        sati=[0,1,2,4,16,25,33,34,35,36,37,50,66,59,67,68]                                           #numeric
        char=[3,11,12,13,14,15,17,23,24,45,46,47,48,49,63,64,65]                                     #alphabetic
        chek=[5,6,7,8,9,10,18,19,20,21,22,27,31,38,39,40,41,42,43,51,52,53,54,55,56,57,58]           #alpha numeric
        boxx=[0,1,2,3,4,11,12,13,14,15,16,17,23,24,25,26,28,29,30,32,33,34,35,36,
              37,44,45,46,47,48,49,50,59,60,61,62,63,64,65,66,67,68,69,70]
        printht=20
        vert=15
    elif joint and inn==2:
        nanu=0.25
        lp=65
        sati=[29,30,31,33,54,57,58,59,60,61,62,64,65,66]
        char=[3,32,40,41,42,43,44,45,46,52,53]
        chek=[0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,34
              ,35,36,37,38,39,47,48,49,50,51,56,63]
        boxx=[29,30,31,33,54,57,58,59,60,61,62,64,65,66,3,32,40,41,42,43,44,45,46,52,53,0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,34
              ,35,36,37,38,39,47,48,49,50,51,56,63,53]
        printht=20
        vert=15
    elif joint and inn==3:
        nanu=0.1
        lp=73
        sati=[12,21,28,29,30,61,68]
        char=[9,10,11,25,26,27,35]
        chek=[0,1,2,3,4,5,13,14,15,16,17,18,19,20,32,33,34,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,
              53,54,55,56,57,58,59,60,62,63,64,65,66,67,69,70,71,72,73]
        boxx=[0,1,2,3,4,5,13,14,15,16,17,18,19,20,32,33,34,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,
              53,54,55,56,57,58,59,60,62,63,64,65,66,67,69,70,71,72,73,9,10,11,25,26,27,35,12,21,28,29,30,61,68
              ,6,78,22,23,24,31]
        printht=20
        vert=15
    elif joint and inn==4:
        nanu=0.84
        lp=21
        sati=[0,4,6,8,9,10,13,14]
        char=[11,12,19,20]
        chek=[1,2,3,4,5,7,15,16,17,18,21]
        boxx=[1,2,3,4,5,7,15,16,17,18,21,11,12,19,20,0,4,6,8,9,10,13,14]
        printht=20
        vert=15
    elif joint and inn==5:
        nanu=9
        lp=13
        sati=[2,8,12]
        char=[3,7,9,13]
        chek=[0,1,4,11]
        boxx=[5,6,10,2,8,12,3,7,9,13,0,1,4,11]
        printht=20
        vert=15
    elif joint and inn==6:
        nanu=11
        lp=8
        sati=[5,6]
        char=[3,4,7,8]
        chek=[1,2]
        boxx=[5,6]
        printht=20
        vert=15
    elif joint and inn==7:
        nanu=5.35
        lp=42
        sati = [0,3, 11, 7,8, 13, 14, 15, 17, 18, 19, 20, 21, 22,23, 25, 29, 30, 31, 32, 33, 40, 41]
        char = [2, 12,  9, 10, 16, 38, 39]
        chek = [1, 4, 5, 6, 24, 25, 26, 27, 28, 34, 35, 36, 37]
        boxx = [0, 7, 25, 29, 30, 31, 32, 33, 41]
        printht=17
        vert=13
    path='jeet'+inn+'.png'
    #print(path)
    image = cv2.imread('../input/' + path)
    #image= cv2.imread(path)
    h,w,c= image.shape
    hfact=1                 
    wfact=1
    if h>4000 and w>3500:
        hfact=h/2000
        wfact=w/1700
    image=image[10:h-10,10:w-10]
    image1=cv2.resize(image,(w//wfact,h//hfact),interpolation = cv2.INTER_AREA)
    cv2.imwrite('../temp-'+inn+'/original.png', image1)
    gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    #Separate the background from the foreground
    bit = cv2.bitwise_not(gray)
    im = np.array(bit)
    h,w=im.shape
    wid=w
    # Normalize and threshold image
    #im = cv2.normalize(im, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    #res, im = cv2.threshold(im, 64, 255, cv2.THRESH_BINARY) 
    
    # Fill everything that is the same colour (black) as top-left corner with white
    cv2.floodFill(im, None, (0,0), 255)
    
    # Fill everything that is the same colour (white) as top-left corner with black
    cv2.floodFill(im, None, (0,0), 0)
    cv2.floodFill(im, None, (0,w), 255)
    cv2.floodFill(im, None, (0,w), 0)
    
    cv2.floodFill(im, None, (w-1,0), 255)
    cv2.floodFill(im, None, (w-1,0), 0)
    
    
    # Save result
    image=np.invert(im)
    cv2.imwrite('../temp-'+inn+'/noblack.png', image)
    
    image=cv2.imread('../temp-'+inn+'/noblack.png')
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)
    print(angle)
    if angle<-45:
        angle+=90
    rotated = rotate(image, angle, (255, 255, 255))               #rotating
    #print(angle)
    orig=rotated
    cv2.imwrite('../temp-'+inn+'/deskewed.png', rotated)
    yen_threshold = threshold_yen(rotated)
    bright = rescale_intensity(rotated, (170, yen_threshold+10), (0, 255))
    cv2.imwrite('../temp-'+inn+'/bright.png',bright)  
    gre=cv2.cvtColor(bright, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gre, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.imwrite('../temp-'+inn+'/binary.png', thresh)
    binar=cv2.imread('../temp-'+inn+'/binary.png')
    ##detecting logo and dark strip using square kernel
    ##To standardize image using logo and dark strip as reference
    image = binar
    result = orig.copy()
    if joint:
        ker=(40,10)
        if inn=='7':
            image=binar
            ker=(50,10)
        elif inn=='1':
            ker=(150,10)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        left=[]
        rgt=[]
        l1=[]
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ker)
        remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        may=5000
        may1=0
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if x+w<wid-20 and x>20:
                cv2.drawContours(result, [c], 0, (255,0,255), -1)
                left.append(x)
                l1.append(x+w)
                rgt.append(y)
                if may>y  and may1<x:
                    may=y
                    mas=x+w
        cv2.imwrite('../temp-'+inn+'/square_kernels.png',result)
        if inn=='5'or '6' :
            ss=min(rgt)
            for i in range(len(rgt)):
                if rgt[i]>1200:
                    rgt[i]=ss
        v=(max(rgt)-min(rgt))*nanu
        v=int(v//1)
        #print(v)
        crop=result[min(rgt):max(rgt)+v,min(left):mas]
        crop_binar=binar[min(rgt):max(rgt)+v,min(left):mas]
        width = 1250                                 #resizing into same dimensions
        height = 1960
        dim = (width, height)
        if inn=='1':
            crop=result[min(rgt)-v-v+(v//2)-15:max(rgt)+v,min(left):max(l1)]
            crop_binar=binar[min(rgt)-v-v+(v//2)-15:max(rgt)+v,min(left):max(l1)]
            print("sssssss",min(rgt),max(rgt),min(left),max(l1))
        if inn=='7':
            abi=[]
            width = 1250                             #resizing into same dimensions
            height = 1260
            dim = (width, height)
            up=min(rgt)
            for i in range(len(rgt)):
                print("ss",rgt[i])
                if rgt[i]>500 and rgt[i]< 1700:
                    #print(rgt[i])
                    abi.append(rgt[i])
                    rgt[i]=0
            #print(rgt)
            if abi == []:
                down=max(rgt)
            else:
                down=min(abi)
                nanu=0
            v=(down-up)*nanu
            v=int(v//1)
            lab=max(left)
            if min(left)<50: 
                for i in range(len(left)):
                    if left[i]<50:
                        left[i]=lab
            crop=result[up:down+v,min(left):max(l1)]
            #print(down+v-up,up,down,v)
            crop_binar=binar[up:down+v,min(left):max(l1)]
        
    else:    
        ker=(40,10)
        if inn=='4':
            image=binar
            ker=(50,10)
        elif inn=='1':
            ker=(150,10)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        left=[]
        rgt=[]
        l1=[]
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ker)
        remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        may=5000
        may1=0
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if x+w<wid-20 and x>20:
                cv2.drawContours(result, [c], 0, (255,0,255), -1)
                left.append(x)
                l1.append(x+w)
                rgt.append(y)
                if may>y  and may1<x:
                    may=y
                    mas=x+w
        cv2.imwrite('../temp-'+inn+'/square_kernels.png',result)
        if inn=='3':
            ss=min(rgt)
            for i in range(len(rgt)):
                if rgt[i]>1200:
                    rgt[i]=ss
        v=(max(rgt)-min(rgt))*nanu
        v=int(v//1)
        #print(v)
        crop=result[min(rgt):max(rgt)+v,min(left):mas]
        crop_binar=binar[min(rgt):max(rgt)+v,min(left):mas]
        width = 1250                                 #resizing into same dimensions
        height = 1960
        dim = (width, height)
        if inn=='1':
            crop=result[min(rgt)-v-v+(v//2)-15:max(rgt)+v,min(left):max(l1)]
            crop_binar=binar[min(rgt)-v-v+(v//2)-15:max(rgt)+v,min(left):max(l1)]
            print("sssssss",min(rgt),max(rgt),min(left),max(l1))
        if inn=='4':
            abi=[]
            width = 1250                             #resizing into same dimensions
            height = 1260
            dim = (width, height)
            up=min(rgt)
            for i in range(len(rgt)):
                print("ss",rgt[i])
                if rgt[i]>500 and rgt[i]< 1700:
                    #print(rgt[i])
                    abi.append(rgt[i])
                    rgt[i]=0
            #print(rgt)
            if abi == []:
                down=max(rgt)
            else:
                down=min(abi)
                nanu=0
            v=(down-up)*nanu
            v=int(v//1)
            lab=max(left)
            if min(left)<50: 
                for i in range(len(left)):
                    if left[i]<50:
                        left[i]=lab
            crop=result[up:down+v,min(left):max(l1)]
            #print(down+v-up,up,down,v)
            crop_binar=binar[up:down+v,min(left):max(l1)]
    # resize image
    x1 = cv2.resize(crop, dim, interpolation = cv2.INTER_AREA)
    x2 = cv2.resize(crop_binar, dim, interpolation  = cv2.INTER_AREA)
    cv2.imwrite('../temp-'+inn+'/cropped.png',x1)  
    cv2.imwrite('../temp-'+inn+'/croppedbinar.png',x2)
    #cropped image
    ##Removing small printed texts using contours and its heights
    # Load image, grayscale, Otsu's threshold
    image =x2
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image=x1
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)[1]
    # Find contours, sort from left-to-right, then crop
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts, _ = contours.sort_contours(cnts, method="left-to-right")
    # Filter using contour area and extract ROI
    ROI_number = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 10000:
            x,y,w,h = cv2.boundingRect(c)
            if h<printht and h>5:
                cv2.rectangle(image, (x-1, y-1), (x + (w+1), y + (h+1)), (255,255,255),-1)
                cv2.rectangle(x2, (x-1, y-1), (x + (w+1), y + (h+1)), (255,255,255),-1)
                ROI_number += 1
    cv2.imwrite('../temp-'+inn+'/'+'noprint.png', image)
    
    ###Removing boxes and logo and dark strip
    result = image.copy()
    image=x2
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cv2.imwrite('../temp-'+inn+'/'+'edge_detected.png', thresh)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,10))
    remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(result, [c], -1, (255,255,255), -1)
        cv2.drawContours(x2, [c], -1, (255,255,255), -1)
    #removes horizontal lines
    
 # ###################################   
#     horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,15))
#     remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
#     cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#     for c in cnts:
#         cv2.drawContours(result, [c], -1, (255,255,255), 1)
        
        
#     repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
#     result = 255 - cv2.morphologyEx(255 - result, cv2.MORPH_CLOSE, repair_kernel, iterations=1)
# # ###########################

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,20))#########
    remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(result, [c], -1, (255,255,255), -1)###########
        cv2.drawContours(x2, [c], -1, (255,255,255), -1)
        #######################################################
    # repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))
    # result = 255 - cv2.morphologyEx(255 - result, cv2.MORPH_CLOSE, repair_kernel, iterations=1)
###############################################################
    cv2.imwrite('../temp-'+inn+'/'+'logo_removed.png', result)


    cv2.imwrite('../temp-'+inn+'/'+'clean.png', result)
    cv2.imwrite('../temp-'+inn+'/'+'clean_bin.png', x2)
    image=result
    x11=cv2.imread('../temp-'+inn+'/'+'clean.png')
    

    ####Extracting lines using XML template
    #Load template
    template_file='../xml_template/p-'+inn+'.xml'

    annotations=template_file
    in_file = open(template_file)
    tree=ET.parse(in_file)
    root = tree.getroot()
    jpg = annotations.split('.')[0] + '.jpg'
    imsize = root.find('size')
    w = int(imsize.find('width').text)
    h = int(imsize.find('height').text)
    all = list()

    for obj in root.iter('object'):
            current = list()
            name = obj.find('name').text
            xmlbox = obj.find('bndbox')
            xn = int(float(xmlbox.find('xmin').text))
            xx = int(float(xmlbox.find('xmax').text))
            yn = int(float(xmlbox.find('ymin').text))
            yx = int(float(xmlbox.find('ymax').text))
            current += [jpg,w,h,name,xn,yn,xx,yx]
            all += [current]

    in_file.close()

    data = pd.DataFrame(all,columns=['path','width','height','label','xmin','ymin','xmax','ymax'])

    # Read input image
    img = image
    xx=x11
    ella=[]
    overlay1 = img.copy()
    for i,row in data.iterrows():
        x1=row['xmin']
        y1=row['ymin']
        x2=row['xmax']
        y2=row['ymax']
        immm=img[y1:y2,x1:x2]
        image=img[y1:y2,x1:x2]                      #cropping fields from whole image
        
        
        #cv2.rectangle(img,(x1,y1),(x2, y2 ),(90,0,25),2)
        crop_image_path='../temp-'+inn+'/lines/R_'+str(i)+'.png'
        image1=removebox(immm,image,boxx,chek,i,yen_threshold)
        cv2.imwrite(crop_image_path,image1)
        ella.append(image1)
        #cv2.imwrite('../temp-'+inn+'maa.png',img)
        cv2.rectangle(overlay1, (x1, y1), (x2, y2), (0, 200, 0), -1)
        # cv2.rectangle(overlay1, (int(x1_min), int(y1_min)), (int(x4_max), int(y4_max)), (0, 200, 0), -1)
        #print('rectangle')
        alpha = 0.3  # Transparency factor.
        # Following line overlays transparent rectangle over the image
        img_new = cv2.addWeighted(overlay1, alpha, img, 1 - alpha, 0)
        #print('new image created')
        r = 1000.0 / img_new.shape[1]  # resizing image without loosing aspect ratio
        dim = (1000, int(img_new.shape[0] * r))
        #print(dim)
        # perform the actual resizing of the image and show it
    resized = cv2.resize(img_new, dim, interpolation=cv2.INTER_AREA)
    #print('resized')
    cv2.imwrite('../mark'+inn+'.png',resized)
    #print("line done")

    
    de,de1=wordseg(lp,sati,char,chek,all,ella)           #passing extracted lines to further steps
    #print(de)
    return de,de1

def removebox(use,orig,boxx,chek,i,yenn):
    if i in chek:
        v_ker=(1,8)
        h_ker=(10,1)
    elif i in boxx:
        v_ker=(1,14)
        h_ker=(16,1)
    else:
        v_ker=(2,50)
        h_ker=(35,1)
    use = rescale_intensity(use, (170, yenn+10), (0, 255))
    #removes vertical line
    gray1 = cv2.cvtColor(use,cv2.COLOR_BGR2GRAY)
    gray=cv2.cvtColor(orig,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, v_ker)
    remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    xp=0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        su=0
        if w==1:
            avg=np.mean(gray[y:y+h,x:x+w])
            mi=np.min(gray[y:y+h,x:x+w])
            # print(avg,mi)
            if abs(avg-mi)<45:
                avg=150
            # print(avg)
            for i in range(y,y+h):
                if gray[i][x]>avg-40:
                    gray[i][x]=255
                else:
                    gray[i][x]=int(avg-30)

        elif w==2:
            avg=np.mean(gray[y:y+h,x:x+1])
            mi=np.min(gray[y:y+h,x:x+w])
            # print(avg,mi)
            if abs(avg-mi)<45:
                avg=150
            for i in range(y,y+h):
                if gray[i][x]>avg-40:
                    gray[i][x]=255
                else:
                    gray[i][x]=int(avg-30)
            for i in range(y,y+h):
                if gray[i][x+1]>avg-40:
                    gray[i][x+1]=255
                else:
                    gray[i][x+1]=int(avg-30)
        elif w==3 and abs(x-xp)>50:
            avg=np.mean(gray[y:y+h,x+1:x+2])
            mi=np.min(gray[y:y+h,x:x+w])
            print(avg,mi)
            if abs(avg-mi)<45:
                avg=150
            for i in range(y,y+h):
                if gray[i][x]>avg-40:
                    gray[i][x]=255
                else:
                    gray[i][x]=int(avg-30)
            for i in range(y,y+h):
                if gray[i][x+1]>avg-40:
                    gray[i][x+1]=255
                else:
                    gray[i][x+1]=int(avg-30)
            for i in range(y,y+h):
                if gray[i][x+2]>avg-40:
                    gray[i][x+2]=255
                else:
                    gray[i][x+2]=int(avg-30)
        # avg=su//h
        # avg+=(255-5*avg)
        # for i in range(y,y+h):
        #     orig[i][x]=orig[i][x]+avg
             
        # if w<3:
        #     cv2.drawContours(orig, [c], -1, (255,255,255), 2)
    #xp=x
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, h_ker)
    remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(gray, [c], -1, (255,255,255), 2)
    img = np.stack((gray,)*3, axis=-1)
    return img



# # Line segmentation

#line segmentation code to extract lines 
def lineseg(image):
    #import image
    #grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    #binary
    ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    #dilation
    kernel = np.ones((10,350), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1) 
    #find contours
    ctrs,hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count=-1
    for  ctr in ctrs:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)

        # Getting ROI
        if h>20:
            count+=1
    pred=[]
    for ctr in ctrs:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        #print(x,y,w,h)
        # Getting ROI
        if h>20:

            ROI = image[y:y+h, x:x+w]
            pred.insert(-1,ROI)
            #cv2.imwrite('line_chopped/ROI_{}.png'.format(ROI_number), ROI)
            #cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,25),2)
    #cv2.imwrite('chopped/ROI_{}.png'.format(ROI_number), ROI)

    return pred




def predictlet(im,n):
    cv2.imwrite('temp.png',im)
    gray=cv2.imread('temp.png',0)
    #plt.imshow(gray)
    (thresh,gray1)=cv2.threshold(gray,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #Resizing input image
    gray1=cv2.resize((255-gray1),(28*n,28))
    li=[]
    gs=0
    for i in range(n):
        #creating Sub-Images for the given image
        sub_image = gray1[0: 28*n, i*28:(i+1)*28]
        img=sub_image.flatten()/255.0
        image=img.reshape(-1,28,28,1)
        result=new_model.predict(image)
        result=result[0]
        papu=np.argmax(result)
        ans=finalPrediction[papu]
        li.append(ans)
        g=round(result[papu]*100,3) 
        gs+=g
    av=gs/n

    li.append(" ")
    return li,av




def predictdig(im,n):
    cv2.imwrite('temp.png',im)
    gray=cv2.imread('temp.png',0)
    #plt.imshow(gray)
    (thresh,gray1)=cv2.threshold(gray,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #Resizing input image
    gray1=cv2.resize((255-gray1),(28*n,28))
    li=[]
    gs=0
    for i in range(n):
        #creating Sub-Images for the given image
        sub_image = gray1[0: 28*n, i*28:(i+1)*28]
        img=sub_image.flatten()/255
        image=img.reshape(-1,28,28,1)
        ress=digi_model.predict(image)
        ress=ress[0]
        answer = np.argmax(ress)
        #display the prediction
        ss=str(answer)
        li.append(ss)
        g=round(ress[answer]*100,3)
        gs+=g
    av=gs/n

    li.append(" ")
    return li,av




def predictchar(im,n):
    cv2.imwrite('temp.png',im)
    gray=cv2.imread('temp.png',0)
    #plt.imshow(gray)
    (thresh,gray1)=cv2.threshold(gray,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #Resizing input image
    gray1=cv2.resize((255-gray1),(28*n,28))
    li=[]
    gs=0
    for i in range(n):
        #creating Sub-Images for the given image
        sub_image = gray1[0: 28*n, i*28:(i+1)*28]
        img=sub_image.flatten()/255.0
        image=img.reshape(-1,28,28,1)
        result=new_model.predict(image)
        result=result[0]
        chopres=result[10:36]
        #print(chopres)
        papu=np.argmax(chopres)
        #print(papu)
        ans=finalPrediction[papu+10]
        li.append(ans)
        g=round(result[papu+10]*100,3) 
        gs+=g
    av=gs/n
    li.append(" ")
    return li,av



# # character segmentation



# Load image, grayscale, Otsu's threshold
#here i'm passing the same image to which i saved earlier and dividing'em into individual characters.
def charseg(image,flag,chai,syg):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Find contours, obtain bounding box, extract and save ROI
    ROI_number = 0
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    li=[]
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 1:
            x,y,w,h = cv2.boundingRect(c)
            if w>40 and w<100 and h>12:
                li.append([x,w//2,y,h])
                li.append([x+w//2,w//2,y,h])
                ROI_number += 2
                if inn=='4':
                    print("pa4",w,syg )
            elif h>10 and w>3:
                li.append([x,w,y,h])
                ROI_number += 1
                    
                    # dodda=image[:,x:x+w]
                    # cv2.imwrite('marked_{}.png'.format(w),dodda)
    les=sorted(li,key=lambda x:x[0])
    if ROI_number==0:
        return [],0,image
    desired_size = 28
    for i in range(ROI_number):
        if flag==1:
            ROI = image[ les[i][2]:les[i][2]+les[i][3] , les[i][0]:les[i][0]+les[i][1]]
        else:
            ROI = image[ les[i][2]:les[i][2]+les[i][3] , les[i][0]:les[i][0]+les[i][1]]
        #cv2.imwrite('chopped1/ltr_{}.png'.format(i), ROI)
        im=ROI
        old_size = im.shape[:2] # old_size is in (height, width) format

        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        # new_size should be in (width, height) format
        #print(old_size)
        #print((new_size[1], new_size[0]))
        im = cv2.resize(im, (new_size[1]+1, new_size[0]))
        if inn=='4':
            im = cv2.resize(im, (new_size[1], new_size[0]))
            
        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        color = [255, 255, 255]
        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
        if i==0:
            vis = new_im
        else:
            vis=np.concatenate((vis,new_im), axis=1)
        #cv2.imwrite('temp/a.png',vis)
    if flag==1:
        ps,ds=predictdig(vis,ROI_number)
        return ps,ds,vis
    if chai==1:
        ps,ds=predictchar(vis,ROI_number)
        return ps,ds,vis
    else:
        ps,ds=predictlet(vis,ROI_number)
        return ps,ds,vis
# # Word segmentation



#word segmentation from line it extracts word by word
def wordseg(lp,sati,char,chek,all,ella):
    dictt={}
    dictt1={}
    str1=""
    f=open('../result/out'+inn+'.txt','w')
    for syg in range(0,lp):
        predlist=[]
        finav=0
        flag=0
        chai=0
        if syg in sati:
            flag=1
        if syg in char:
            chai=1
        # Load the image
        #img = cv2.imread('../temp-'+inn+'/lines/R_{}.png'.format(syg))
        img=ella[syg]
        #lo=lineseg(img)
        for j in range(1):
            gg=img
            #cv2.imwrite('cropped/lines/ROI_{}_{}.png'.format(syg,j), gg)
            img=gg
            lis=[]
            ret1,img = cv2.threshold(img,200,255,cv2.THRESH_BINARY_INV)
            # convert to grayscale
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # smooth the image to avoid noises
            #gray = cv2.medianBlur(gray,5)

            # Apply adaptive threshold
            thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
            #thresh_color = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)

            # apply some dilation and erosion to join the gaps - change iteration to detect more or less area's
            thresh = cv2.dilate(thresh,None,iterations = 5)
            thresh = cv2.erode(thresh,None,iterations = 5)

            # Find the contours
            contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            # For each contour, find the bounding rectangle and draw it
            num=1
            predlist=[]
            for cnt in contours:
                x,y,w,h = cv2.boundingRect(cnt)
                if h>20 and w>20:
                    lis.append([x,w,y,h])
                    num+=1
            lest=sorted(lis,key=lambda x:x[0])
            #print(num,syg)
            for j in range(num-1):
                #print(syg,j)
                ROI=gg[:,lest[j][0]:lest[j][0]+lest[j][1]]
                yammu,av,mm=charseg(ROI,flag,chai,syg)
                finav+=av
                predlist=predlist+yammu
                #cv2.rectangle(gg,(lest[j][0]-1,lest[j][2]),( lest[j][0] + lest[j][1]+2, lest[j][2] + lest[j][3] ),(90,0,255),2)
                cv2.imwrite('../temp-'+inn+'/words/marked_{}_{}.png'.format(syg,j),mm)
            finav/=(num)
        if syg in chek and predlist !=[]:
            predlist=['True']
        if True:
            #print(all[syg][3],end=" : ")
            f.write(all[syg][3])
            f.write(" : ")
            #print(*predlist)
            asd=str1.join(predlist)
            dictt[all[syg][3]]=asd      #######  asd is prediction result, 
            dictt1[all[syg][3]]=finav   ####finav is prediction probability
            for ele in predlist:
                f.write(ele)
            f.write("\n\n")
    f.close()
    return dictt,dictt1



#Loading the model
new_model = tf.keras.models.load_model('../models/merge2.h5')
digi_model=tf.keras.models.load_model('../models/newdigi.h5')
finalPrediction = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","a","6","d","e","f","g","h","n","q","r","t"]

#input page number
lis=[]
problist=[]
joint=0
if joint:
    pages=8
else:
    pages=5
for i in range(1,pages):             #reading image from 1 to 4
    inn=str(i)
    dd,dd1=preproc(inn,joint)          #storing all the dictionary output in a list
    problist.append(dd1)
    lis.append(dd)               #all dictionaries are appended in a list  index+1 is the page number
print("results are stored in respective folder and as a dictionary at the end of code")
