import cv2
import numpy as np
from csi_camera import CSI_Camera
import calibration
from PIL import Image


imgr = cv2.imread('/home/thomas/Desktop/Python_programs/SeniorProject/images/stereoRight/imageR12.png')
imgl = cv2.imread('/home/thomas/Desktop/Python_programs/SeniorProject/images/stereoLeft/imageL12.png')

rect_right, rect_left = calibration.undistortRectify(imgr, imgl)


cv2.imwrite('/home/thomas/Desktop/Python_programs/SeniorProject/images/stereoLeft/imagerectL.png', rect_left)
cv2.imwrite('/home/thomas/Desktop/Python_programs/SeniorProject/images/stereoRight/imagerectR.png', rect_right)