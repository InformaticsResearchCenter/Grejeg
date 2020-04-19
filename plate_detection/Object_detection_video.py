#import library
import numpy as np
import cv2
from detector import detector
from imutils import paths
import os
import time

#import plot_cv dari folder utils
from utils import plot_cv

#untuk notifikasi 
print("[INFO] starting video stream...")

#untuk mengatur path pada file yang akan di test
vs = cv2.VideoCapture('test_image/testing_vidio.mp4')
#melakukan jeda
time.sleep(2.0)

#variabel det untuk memanggil class detector
det = detector.detector()

#nama class
name_class = ["plate"]

# Mengulang setiap gambar
while True:

    #print(imageName)
    ret, image = vs.read()
    #untuk mendetek
    (boxes, scores, classes, num, category_index) = det.detect_plate(image)
    #jika box tidak ada maka next
    if len(boxes) == 0:
        print('skip detection')
        continue
    #ukuran gambar original
    h, w = image.shape[:2]
    boxes = boxes[0]
    scores = scores[0]
    classes = classes[0]

    #untuk filter box mana yang akan ditampilan
    for box, score, class_idx in zip(boxes, scores, classes):
        (startY, startX, endY, endX) = box
        #koordinat box dikali dengan lebar dan tinggi gambar
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
  
    # Semua hasil telah diambil pada gambar. Sekarang tampilkan gambar.
    cv2.namedWindow("Plate detector", cv2.WINDOW_KEEPRATIO)
    cv2.setWindowProperty("Plate detector", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    cv2.imshow('Plate detector', image)

    # tekan apapun untuk menutup gambar
    key = cv2.waitKey(1) & 0xFF

    # jika menekan q maka proses loop berhenti
    if key == ord("q"):
        break
# Menutup window
cv2.destroyAllWindows()
