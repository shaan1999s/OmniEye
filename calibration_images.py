import cv2
import sterio_vision_control as svc
import numpy as np
from PIL import Image
'''
store the images for the sterio calibration of the sterio camera setup
'''
show_fps = True

left_camera = svc.create_camera(1)
right_camera = svc.create_camera(0)

cv2.namedWindow("CSI Cameras", cv2.WINDOW_AUTOSIZE)
num = 0
if(
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
    while cv2.getWindowProperty("CSI Cameras", 0) >= 0:
        key = cv2.waitKey(25)

        left_image = svc.read_camera(left_camera, show_fps)
        right_image = svc.read_camera(right_camera, show_fps)

        # left_image = Image.fromarray(left_image)
        # right_image = Image.fromarray(right_image)

      
        # We place both images side by side to show in the window
        if key == ord('s'):  # wait for 's' key to save and exit
            left_file = "/home/thomas/Desktop/Python_programs/SeniorProject/images/stereoLeft/imageL" + str(num) + '.png'
            right_file = "/home/thomas/Desktop/Python_programs/SeniorProject/images/stereoRight/imageR" + str(num) + '.png'
            # # Image.open(left_file)
            # # Image.open(left_file)
            # # open(left_file, 'x')
            # # open(right_file, 'x')
           

            # left_image.save(left_file,'png')
            # right_image.save(right_file,'png')
            cv2.imwrite('/home/thomas/Desktop/Python_programs/SeniorProject/images/stereoLeft/imageL' + str(num) + '.png', left_image)
            cv2.imwrite('/home/thomas/Desktop/Python_programs/SeniorProject/images/stereoRight/imageR' + str(num) + '.png', right_image)
            print("images saved!")

            num += 1
        camera_images = np.hstack((right_image, left_image))
        resized_images = cv2.resize(camera_images, (1640,616), interpolation = cv2.INTER_AREA)
        cv2.imshow("CSI Cameras",resized_images)
        left_camera.frames_displayed += 1
        right_camera.frames_displayed += 1
        # This also acts as a frame limiter
        # Stop the program on the ESC key
        if (key & 0xFF) == 27:
            break

finally:
    left_camera.stop()
    left_camera.release()
    right_camera.stop()
    right_camera.release()
cv2.destroyAllWindows()


# cap = cv2.VideoCapture(0)
# cap2 = cv2.VideoCapture(2)

# num = 0


# while cap.isOpened():

#     succes1, img = cap.read()
#     succes2, img2 = cap2.read()

#     k = cv2.waitKey(5)

#     if k == 27:
#         break
#     elif k == ord('s'):  # wait for 's' key to save and exit
#         cv2.imwrite('images/stereoLeft/imageL' + str(num) + '.png', img)
#         cv2.imwrite('images/stereoright/imageR' + str(num) + '.png', img2)
#         print("images saved!")
#         num += 1

#     cv2.imshow('Img 1', img)
#     cv2.imshow('Img 2', img2)
