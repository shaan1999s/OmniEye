import numpy as np
import cv2 as cv
import glob



################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (9,6)
# frameSize = (640,480)
frameSize = (1640,1232)
#

calibration_flags = cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv.fisheye.CALIB_CHECK_COND+cv.fisheye.CALIB_FIX_SKEW

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
# objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

objp = np.zeros((chessboardSize[0]*chessboardSize[1], 1, 3), np.float64)
objp[:,0, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
imgpointsR = [] # 2d points in image plane.


imagesLeft = glob.glob('/home/thomas/Desktop/Python_programs/SeniorProject/images/stereoLeft/*.png')
imagesRight = glob.glob('/home/thomas/Desktop/Python_programs/SeniorProject/images/stereoRight/*.png')
print(len(imagesRight))
# imagesLeft = glob.glob('images/stereoLeft/*.png')
# imagesRight = glob.glob('images/stereoRight/*.png')
imagesLeft.sort()
imagesRight.sort() 

num = 0

for imgLeft, imgRight in zip(imagesLeft, imagesRight):

    imgL = cv.imread(imgLeft)
    imgR = cv.imread(imgRight)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
    retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if retL and retR == True:

        objpoints.append(objp)

        cornersL = cv.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
        imgpointsL.append(cornersL)

        cornersR = cv.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
        imgpointsR.append(cornersR)

        # Draw and display the corners
        cv.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
        cv.imshow('img left', imgL)
        cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
        cv.imshow('img right', imgR)

        cv.imwrite('/home/thomas/Desktop/Python_programs/SeniorProject/images/stereoLeft/imageLCB' + str(num) + '.png', imgL)
        cv.imwrite('/home/thomas/Desktop/Python_programs/SeniorProject/images/stereoRight/imageRCB' + str(num) + '.png', imgR)
        num += 1
        cv.waitKey(100)
    else:
        print("chessboard detection falied on:", imgLeft, "and ", imgRight)
        continue

cv.destroyAllWindows()

N_OK = len(objpoints)
DIM = (1640, 1232)

K_l = np.zeros((3, 3))
D_l = np.zeros((4, 1))
rvecs_l = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs_l = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

K_r = np.zeros((3, 3))
D_r = np.zeros((4, 1))
rvecs_r = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs_r= [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

# Single camera calibration (undistortion)
rms_l, camera_matrix_l, distortion_coeff_l, _, _ = cv.fisheye.calibrate(objpoints, imgpointsL, grayL.shape[::-1], K_l, D_l, rvecs_l, tvecs_l, calibration_flags, (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
# Let's rectify our results
map1_l, map2_l = cv.fisheye.initUndistortRectifyMap(K_l, D_l, np.eye(3), K_l, DIM, cv.CV_16SC2)

rms_r, camera_matrix_r, distortion_coeff_r, _, _ = cv.fisheye.calibrate(objpoints, imgpointsR, grayR.shape[::-1], K_r, D_r, rvecs_r, tvecs_r, calibration_flags, (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
# Let's rectify our results
map1_r, map2_r = cv.fisheye.initUndistortRectifyMap(K_r, D_r, np.eye(3), K_r, DIM, cv.CV_16SC2)

TERMINATION_CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.01)
OPTIMIZE_ALPHA = 0.25

imgpointsL = np.asarray(imgpointsL, dtype=np.float64)
imgpointsR = np.asarray(imgpointsR, dtype=np.float64)


(RMS, _, _, _, _, rotationMatrix, translationVector) = cv.fisheye.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR,
    camera_matrix_l, distortion_coeff_l,
    camera_matrix_r, distortion_coeff_r,
    DIM, None, None,
    cv.CALIB_FIX_INTRINSIC, TERMINATION_CRITERIA)

R1 = np.zeros([3, 3])
R2 = np.zeros([3, 3])
P1 = np.zeros([3, 4])
P2 = np.zeros([3, 4])
Q = np.zeros([4, 4])

# Rectify calibration results
(leftRectification, rightRectification, leftProjection, rightProjection,
 dispartityToDepthMap) = cv.fisheye.stereoRectify(
    camera_matrix_l, distortion_coeff_l,
    camera_matrix_r, distortion_coeff_r,
    DIM, rotationMatrix, translationVector,
    0, R2, P1, P2, Q,
    cv.CALIB_ZERO_DISPARITY, (0, 0), 0, 0)

stereoMapL = cv.fisheye.initUndistortRectifyMap(
    camera_matrix_l, distortion_coeff_l, leftRectification,
    leftProjection, DIM, cv.CV_16SC2)

stereoMapR = cv.fisheye.initUndistortRectifyMap(
    camera_matrix_r, distortion_coeff_r, rightRectification,
    rightProjection, DIM, cv.CV_16SC2)

print("Saving parameters!")
cv_file = cv.FileStorage('/home/thomas/Desktop/Python_programs/SeniorProject/stereoMap.xml', cv.FILE_STORAGE_WRITE)


cv_file.write('stereoMapL_x',stereoMapL[0])
cv_file.write('stereoMapL_y',stereoMapL[1])
cv_file.write('stereoMapR_x',stereoMapR[0])
cv_file.write('stereoMapR_y',stereoMapR[1])

cv_file.release()