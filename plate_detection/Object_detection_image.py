#import library
import numpy as np
import cv2
from detector import detector
from imutils import paths
import os
import time

#import plot_cv dari folder utils
from utils import plot_cv

#variabel det untuk memanggil class detector
det = detector.detector()

#mengatur path lukasi image yang akan di testing
image_folder = 'test_image'

#menyimpan file test image
imageNames = os.listdir(image_folder)

#nama class  
name_class = ["plate"]

# mengulang pada setiap image yang ada pada folder 
for imageName in imageNames:

    #membaca file gambar
    image = cv2.imread(image_folder + '/' + imageName)
    
    #untuk ngedetek lokasi kotak pada gambar 
    (boxes, scores, classes, num, dump ) = det.detect_plate(image)

    #jika dalam gambar tidak ditemukan box maka next
    if len(boxes) == 0:
        print('skip detection')
        continue 

    #get ukuran gambar original
    h, w = image.shape[:2] 
    boxes = boxes[0]
    scores = scores[0]
    classes = classes[0]

    #untuk filter box mana yang akan ditampilan
    for box, score, class_idx in zip(boxes, scores, classes):
        (startY, startX, endY, endX) = box
        #koordinat box
        startX = int(startX * w)
        startY = int(startY * h)
        endX = int(endX * w)
        endY = int(endY * h)
        #koordinat box yang di dapat
        box = (startX, startY, endX, endY)
        #untuk menampilkan persentase confidence rate dalam %
        text = name_class[int(class_idx-1)] + " " + str(int(score * 100)) + "%"
        #jika score lebih besar dari 0,7 maka box akan ditampilkan
        if score > 0.7:
            image = plot_cv.plot_object(image, box, text, int(class_idx-1))

    #  Semua hasil telah diambil pada gambar. Sekarang tampilkan gambar.
    cv2.namedWindow("plate detector", cv2.WINDOW_KEEPRATIO)
    cv2.setWindowProperty("plate detector", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    cv2.imshow('plate detector', image)

    # tekan apapun untuk menutup gambar
    cv2.waitKey(0)

# Menutup window
cv2.destroyAllWindows()
