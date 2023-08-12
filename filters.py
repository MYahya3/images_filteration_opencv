import cv2
import numpy as np
import streamlit as st

def bw_filter(img):
    img_gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    return img_gray

def BrownEffect(img):
        # Converting to RGB as sepia matrix below is for RGB.
        img_sepia = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_sepia = np.array(img_sepia, dtype=np.float64)
        img_sepia = cv2.transform(img_sepia, np.matrix([[0.393, 0.769, 0.189],
                                                        [0.349, 0.686, 0.168],
                                                        [0.272, 0.534, 0.131]]))
        # Clip values to the range [0, 255].
        img_sepia = np.clip(img_sepia, 0, 255)
        img_sepia = np.array(img_sepia, dtype=np.uint8)
        img_sepia = cv2.cvtColor(img_sepia, cv2.COLOR_RGB2BGR)
        return img_sepia

def CannyEdgeD(img, thres_1 = 100, thres_2=200):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = cv2.Canny(img_gray, thres_1, thres_2)
    img = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    return img

def pencil_sketch(img, ksize=5):
    img_blur = cv2.GaussianBlur(img, (ksize, ksize), 0, 0)
    img_sketch, _ = cv2.pencilSketch(img_blur)
    return img_sketch

def stylization(img, k=5):
    img_blur = cv2.GaussianBlur(img, (k,k), 0, 0)
    img_style = cv2.stylization(img_blur, sigma_s = 40, sigma_r = 0.1)
    return  img_style

def sharpen(img, k=9):
    kernel = k * np.array([[-1, -1, -1],
                       [-1,  9.5, -1],
                       [-1, -1, -1]])

    img_sharpen = cv2.filter2D(img, -1, kernel)
    return img_sharpen