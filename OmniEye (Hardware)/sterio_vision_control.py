# MIT License
# Copyright (c) 2019,2020 JetsonHacks
# See license
# A very simple code snippet
# Using two  CSI cameras (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit (Rev B01) using OpenCV
# Drivers for the camera and OpenCV are included in the base image in JetPack 4.3+

# This script will open a window and place the camera stream from each camera in a window
# arranged horizontally.
# The camera streams are each read in their own thread, as when done sequentially there
# is a noticeable lag

import cv2
import numpy as np
from csi_camera import CSI_Camera
import calibration
from PIL import Image

show_fps = True

# Simple draw label on an image; in our case, the video frame
def draw_label(cv_image, label_text, label_position):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    color = (255,255,255)
    # You can get the size of the string with cv2.getTextSize here
    cv2.putText(cv_image, label_text, label_position, font_face, scale, color, 1, cv2.LINE_AA)

# Read a frame from the camera, and draw the FPS on the image if desired
# Return an image
def read_camera(csi_camera,display_fps):
    _ , camera_image=csi_camera.read()
    if display_fps:
        draw_label(camera_image, "Frames Displayed (PS): "+str(csi_camera.last_frames_displayed),(90,100))
        draw_label(camera_image, "Frames Read (PS): "+str(csi_camera.last_frames_read),(90,120))
    return camera_image

# Good for 1280x720
# DISPLAY_WIDTH=640
# DISPLAY_HEIGHT=360
# For 1920x1080
# DISPLAY_WIDTH=1280
# DISPLAY_HEIGHT=720
# for 1640x1232
DISPLAY_WIDTH=1640
DISPLAY_HEIGHT=1232

# 1920x1080, 30 fps
SENSOR_MODE_1080=2
# 1280x720, 60 fps
SENSOR_MODE_720=4
# 1640x1232, 30fps
SENSOR_MODE_1232=3

'''
creates a camera instance and a seperate thread for the camera
'''
def create_camera(right_or_left):
    camera = CSI_Camera()
    camera.create_gstreamer_pipeline(
            sensor_id=right_or_left,
            sensor_mode=SENSOR_MODE_1232,
            framerate=30,
            flip_method=0,
            display_height=DISPLAY_HEIGHT,
            display_width=DISPLAY_WIDTH,
    )
    camera.open(camera.gstreamer_pipeline)
    camera.start()
    return camera

def turn_on_cameras():
    left_camera = create_camera(1)
    right_camera = create_camera(0)
    return left_camera, right_camera

def turn_on_sterio_vision():
    left_camera = create_camera(1)
    right_camera = create_camera(0)

    cv2.namedWindow("CSI Cameras", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("undistorted and rectified", cv2.WINDOW_AUTOSIZE)

    if (
        not left_camera.video_capture.isOpened()
        or not right_camera.video_capture.isOpened()
    ):
        # Cameras did not open, or no camera attached

        print("Unable to open any cameras")
        # TODO: Proper Cleanup
        SystemExit(0)
    try:
        # Start counting the number of frames read and displayed
        left_camera.start_counting_fps()
        right_camera.start_counting_fps()
        while cv2.getWindowProperty("CSI Cameras", 0) >= 0 :
            left_image=read_camera(left_camera,show_fps)
            right_image=read_camera(right_camera,show_fps)
            # left_image=cv2.resize(left_image, (1280, 720),interpolation = cv2.INTER_AREA)
            # right_image=cv2.resize(right_image, (1280, 720),interpolation = cv2.INTER_AREA)
            # left_image = Image.fromarray(left_image)
            # right_image = Image.fromarray(right_image)

            rect_right, rect_left = calibration.undistortRectify(right_image, left_image)
            rect_images = np.hstack((rect_right, rect_left))
            rect_images =  cv2.resize(rect_images, (1920,540), interpolation = cv2.INTER_AREA)
            # We place both images side by side to show in the window
            # camera_images = np.hstack((left_image, right_image))
            # camera_images =  cv2.resize(camera_images, (1640,616), interpolation = cv2.INTER_AREA)
            # cv2.imshow("CSI Cameras", camera_images)
            cv2.imshow("undistorted and rectified",rect_images)
            left_camera.frames_displayed += 1
            right_camera.frames_displayed += 1
            # This also acts as a frame limiter
            # Stop the program on the ESC key
            if (cv2.waitKey(10) & 0xFF) == 27:
                break   

    finally:
        left_camera.stop()
        left_camera.release()
        right_camera.stop()
        right_camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    turn_on_sterio_vision()
