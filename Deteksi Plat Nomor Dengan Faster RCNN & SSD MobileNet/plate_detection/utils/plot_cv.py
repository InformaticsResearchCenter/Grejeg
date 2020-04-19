#import library
import cv2
import datetime
import numpy as np

#var ploot_object berisi frame, box, text dan warna
def plot_object(frame, box, text, idx_color):
    #untuk warna RGB
    color_bgr = (0, 147, 0)
    overlay = frame.copy()
    #ukuran Gambar 
    (H, W) = frame.shape[:2]
    #untuk ketebalan box , w gambar dibagi 400
    box_border = int(W / 400)
    #koordinat box
    (startX, startY, endX, endY) = box
    y = startY - 10 if startY - 10 > 10 else startY + 10
    yBox = y + 5
    #index warna hijau
    cv2.rectangle(overlay, (startX, startY), (endX, endY),
                  (255, 255, 255), box_border+4)
    #untuk kotak text
    cv2.rectangle(overlay, (startX, startY), (endX, endY),
                  color_bgr, box_border+2)
   
    font = cv2.FONT_HERSHEY_SIMPLEX

    # make a black image
    img = np.zeros((500, 500))
    # set some text
    # get the width and height of the text box
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=(0.4*box_border), thickness=box_border)[0]
    # set the text start position
    text_offset_x = startX
    text_offset_y = y
    # make the coords of the box with a small padding of two pixels
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
    cv2.rectangle(overlay, box_coords[0], box_coords[1], color_bgr, cv2.FILLED)
    cv2.putText(overlay, text, (text_offset_x, text_offset_y), font, fontScale=0.70, color=(255, 255, 255), thickness=2)


    alpha = 1  # Transparency factor.

    # Following line overlays transparent rectangle over the image
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    return frame