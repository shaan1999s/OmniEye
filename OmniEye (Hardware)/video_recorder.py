import sterio_vision_control as svc
import numpy as np
import cv2
import calibration

def record_video(file_name):
    cv2.namedWindow("CSI Cameras", cv2.WINDOW_AUTOSIZE)
    show_fps = True
    main_directory = '/home/thomas/Desktop/video_files'
    video_name = file_name
    right_file_path = main_directory + '/right_videos/' + video_name
    left_file_path = main_directory + '/left_videos/' + video_name


    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_left = cv2.VideoWriter(left_file_path, fourcc, 30, (1640, 1232))
    video_right = cv2.VideoWriter(right_file_path, fourcc, 30, (1640, 1232))
    left_camera, right_camera = svc.turn_on_cameras()
    recording = False
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
            left_image=svc.read_camera(left_camera,show_fps)
            right_image=svc.read_camera(right_camera,show_fps)

            #right_image, left_image = calibration.undistortRectify(right_image, left_image)

            # We place both images side by side to show in the window
            camera_images = np.hstack((left_image, right_image))
            camera_images =  cv2.resize(camera_images, (1640,616), interpolation = cv2.INTER_AREA)
            cv2.imshow("CSI Cameras", camera_images)
            left_camera.frames_displayed += 1
            right_camera.frames_displayed += 1
            # This also acts as a frame limiter
            # Stop the program on the ESC key
            if (cv2.waitKey(20) & 0xFF) == 27:
                break  
            # if cv2.waitKey(10) & 0xFF == ord('r'):
            #     recording = True
            # if cv2.waitKey(10) & 0xFF == ord('s'):
            #     recording = False
            #     print('done recording')
            # if recording:
            #     video_left.write(left_image)
            #     video_right.write(right_image)
            #     print('recording')
            video_left.write(left_image)
            video_right.write(right_image)
           # print('recording')

    finally:
        left_camera.stop()
        left_camera.release()
        right_camera.stop()
        right_camera.release()
        video_left.release()
        video_right.release()
    cv2.destroyAllWindows()

def playback_video(filename):
    main_directory = '/home/thomas/Desktop/video_files'
    left_path = main_directory + '/left_videos/' + filename
    right_path = main_directory + '/right_videos/' + filename
    left_capture = cv2.VideoCapture(left_path)
    right_capture = cv2.VideoCapture(right_path)

    #check if successfully opened files
    if (left_capture.isOpened() == False) or (right_capture.isOpened() == False):
        print('failed to open video file')
    frames = 0
    #read video files and display them
    while (left_capture.isOpened() and right_capture.isOpened()):
        success_l, frame_l = left_capture.read()
        success_r, frame_r = right_capture.read()

        if success_l and success_r:
            frames+=1
            # cv2.imshow('left', frame_l)
            # cv2.imshow('right', frame_r)
            camera_images = np.hstack((frame_l, frame_r))
            camera_images =  cv2.resize(camera_images, (1640,616), interpolation = cv2.INTER_AREA)
            cv2.imshow("CSI Cameras", camera_images)

            if cv2.waitKey(90) & 0xFF == ord('q'):
                break
        else:
            print("total frames: " + str(frames))
            print('no frames')
            break
    left_capture.release()
    right_capture.release()
    cv2.destroyAllWindows()

command = ''
print('enter a command')
while command != 'q':
    command = input()
    if command == 'r':
        print('recording, press esc to stop')
        record_video('test.avi')
    if command == 'p':
        print('playing video')
        playback_video('pose_test.avi')
    command == 'a'

# record_video('test.avi')
# playback_video('test.avi')