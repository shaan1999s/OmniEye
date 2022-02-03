
import json
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
from torch2trt import TRTModule
import time, sys
import cv2
import PIL.Image, PIL.ImageDraw, PIL.ImageFont
import numpy as np

import torchvision.transforms as transforms
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import argparse
import os.path
import calibration
# import video_recorder

ani_frame = 0

def draw_keypoints(img, key):
    thickness = 5
    w, h = img.size
    draw = PIL.ImageDraw.Draw(img)
    #draw Rankle -> RKnee (16-> 14)
    if all(key[16]) and all(key[14]):
        draw.line([ round(key[16][2] * w), round(key[16][1] * h), round(key[14][2] * w), round(key[14][1] * h)],width = thickness, fill=(51,51,204))
    #draw RKnee -> Rhip (14-> 12)
    if all(key[14]) and all(key[12]):
        draw.line([ round(key[14][2] * w), round(key[14][1] * h), round(key[12][2] * w), round(key[12][1] * h)],width = thickness, fill=(51,51,204))
    #draw Rhip -> Lhip (12-> 11)
    if all(key[12]) and all(key[11]):
        draw.line([ round(key[12][2] * w), round(key[12][1] * h), round(key[11][2] * w), round(key[11][1] * h)],width = thickness, fill=(51,51,204))
    #draw Lhip -> Lknee (11-> 13)
    if all(key[11]) and all(key[13]):
        draw.line([ round(key[11][2] * w), round(key[11][1] * h), round(key[13][2] * w), round(key[13][1] * h)],width = thickness, fill=(51,51,204))
    #draw Lknee -> Lankle (13-> 15)
    if all(key[13]) and all(key[15]):
        draw.line([ round(key[13][2] * w), round(key[13][1] * h), round(key[15][2] * w), round(key[15][1] * h)],width = thickness, fill=(51,51,204))

    #draw Rwrist -> Relbow (10-> 8)
    if all(key[10]) and all(key[8]):
        draw.line([ round(key[10][2] * w), round(key[10][1] * h), round(key[8][2] * w), round(key[8][1] * h)],width = thickness, fill=(255,255,51))
    #draw Relbow -> Rshoulder (8-> 6)
    if all(key[8]) and all(key[6]):
        draw.line([ round(key[8][2] * w), round(key[8][1] * h), round(key[6][2] * w), round(key[6][1] * h)],width = thickness, fill=(255,255,51))
    #draw Rshoulder -> Lshoulder (6-> 5)
    if all(key[6]) and all(key[5]):
        draw.line([ round(key[6][2] * w), round(key[6][1] * h), round(key[5][2] * w), round(key[5][1] * h)],width = thickness, fill=(255,255,0))
    #draw Lshoulder -> Lelbow (5-> 7)
    if all(key[5]) and all(key[7]):
        draw.line([ round(key[5][2] * w), round(key[5][1] * h), round(key[7][2] * w), round(key[7][1] * h)],width = thickness, fill=(51,255,51))
    #draw Lelbow -> Lwrist (7-> 9)
    if all(key[7]) and all(key[9]):
        draw.line([ round(key[7][2] * w), round(key[7][1] * h), round(key[9][2] * w), round(key[9][1] * h)],width = thickness, fill=(51,255,51))

    #draw Rshoulder -> RHip (6-> 12)
    if all(key[6]) and all(key[12]):
        draw.line([ round(key[6][2] * w), round(key[6][1] * h), round(key[12][2] * w), round(key[12][1] * h)],width = thickness, fill=(153,0,51))
    #draw Lshoulder -> LHip (5-> 11)
    if all(key[5]) and all(key[11]):
        draw.line([ round(key[5][2] * w), round(key[5][1] * h), round(key[11][2] * w), round(key[11][1] * h)],width = thickness, fill=(153,0,51))


    #draw nose -> Reye (0-> 2)
    if all(key[0][1:]) and all(key[2]):
        draw.line([ round(key[0][2] * w), round(key[0][1] * h), round(key[2][2] * w), round(key[2][1] * h)],width = thickness, fill=(219,0,219))

    #draw Reye -> Rear (2-> 4)
    if all(key[2]) and all(key[4]):
        draw.line([ round(key[2][2] * w), round(key[2][1] * h), round(key[4][2] * w), round(key[4][1] * h)],width = thickness, fill=(219,0,219))

    #draw nose -> Leye (0-> 1)
    if all(key[0][1:]) and all(key[1]):
        draw.line([ round(key[0][2] * w), round(key[0][1] * h), round(key[1][2] * w), round(key[1][1] * h)],width = thickness, fill=(219,0,219))

    #draw Leye -> Lear (1-> 3)
    if all(key[1]) and all(key[3]):
        draw.line([ round(key[1][2] * w), round(key[1][1] * h), round(key[3][2] * w), round(key[3][1] * h)],width = thickness, fill=(219,0,219))

    #draw nose -> neck (0-> 17)
    if all(key[0][1:]) and all(key[17]):
        draw.line([ round(key[0][2] * w), round(key[0][1] * h), round(key[17][2] * w), round(key[17][1] * h)],width = thickness, fill=(255,255,0))
    return img

'''
hnum: 0 based human index
kpoint : keypoints (float type range : 0.0 ~ 1.0 ==> later multiply by image width, height
'''
def get_keypoint(humans, hnum, peaks):
    #check invalid human index
    kpoint = []
    human = humans[0][hnum]
    C = human.shape[0]
    for j in range(C):
        k = int(human[j])
        if k >= 0:
            peak = peaks[0][j][k]   # peak[1]:width, peak[0]:height
            peak = (j, float(peak[0]), float(peak[1]))
            kpoint.append(peak)
            #print('index:%d : success [%5.3f, %5.3f]'%(j, peak[1], peak[2]) )
        else:    
            peak = (j, None, None)
            kpoint.append(peak)
            #print('index:%d : None %d'%(j, k) )
   # print(kpoint)

    return kpoint


parser = argparse.ArgumentParser(description='TensorRT pose estimation run')
parser.add_argument('--model', type=str, default='densenet', help = 'resnet or densenet' )
#parser.add_argument('--video', type=str, default='/home/spypiggy/src/test_images/video.avi', help = 'video file name' )
args = parser.parse_args()

with open('/home/thomas/torch2trt/trt_pose/tasks/human_pose/human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)
num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])


if 'resnet' in args.model:
    print('------ model = resnet--------')
    MODEL_WEIGHTS = '/home/thomas/torch2trt/trt_pose/tasks/human_pose/resnet18_baseline_att_224x224_A_epoch_249.pth'
    OPTIMIZED_MODEL = '/home/thomas/torch2trt/trt_pose/tasks/human_pose/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
    model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
    WIDTH = 224
    HEIGHT = 224

else:    
    print('------ model = densenet--------')
    MODEL_WEIGHTS = '/home/thomas/torch2trt/trt_pose/tasks/human_pose/densenet121_baseline_att_256x256_B_epoch_160.pth'
    OPTIMIZED_MODEL = '/home/thomas/torch2trt/trt_pose/tasks/human_pose/densenet121_baseline_att_256x256_B_epoch_160_trt.pth'
    model = trt_pose.models.densenet121_baseline_att(num_parts, 2 * num_links).cuda().eval()
    WIDTH = 256
    HEIGHT = 256

data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

if os.path.exists(OPTIMIZED_MODEL) == False:
    print('-- Converting TensorRT models. This may takes several minutes...')
    model.load_state_dict(torch.load(MODEL_WEIGHTS))
    model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
    torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

t0 = time.time()
torch.cuda.current_stream().synchronize()
for i in range(50):
    y = model_trt(data)
torch.cuda.current_stream().synchronize()
t1 = time.time()

print(50.0 / (t1 - t0))

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')
parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

def get_pose_data(img):
    ### perform body pose estimation
    data = preprocess(img)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    return counts, objects, peaks
    
def create_animation_file(video):
    return

def create_pose_video(video_name, save_data):
    main_directory = '/home/thomas/Desktop/video_files'
    left_path = main_directory + '/left_videos/' + video_name
    right_path = main_directory + '/right_videos/' + video_name
    left_capture = cv2.VideoCapture(left_path)
    right_capture = cv2.VideoCapture(right_path)

    left_pose_path = main_directory + '/left_videos/' + 'pose_' + video_name
    right_pose_path = main_directory + '/right_videos/' + 'pose_' + video_name
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_left = cv2.VideoWriter(left_pose_path, fourcc, 30, (1640, 1232))
    video_right = cv2.VideoWriter(right_pose_path, fourcc, 30, (1640, 1232))

    #check if successfully opened files
    if (left_capture.isOpened() == False) or (right_capture.isOpened() == False):
        print('failed to open video file')
    
    fails = 0
    frames = 0

    #read video files and display them
    while (left_capture.isOpened() and right_capture.isOpened()):
        success_l, frame_l = left_capture.read()
        success_r, frame_r = right_capture.read()

        if success_l and success_r:

            frame_r, frame_l = calibration.undistortRectify(frame_r, frame_l)

            img_l = cv2.resize(frame_l, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            img_r = cv2.resize(frame_r, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

            frame_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2RGB)
            frame_l  = PIL.Image.fromarray(frame_l)

            frame_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2RGB)
            frame_r  = PIL.Image.fromarray(frame_r)

            counts_1, objects_1, peaks_1 = get_pose_data(img_l)
            counts_2, objects_2, peaks_2 = get_pose_data(img_r)

            both_keypoints = []

            for i in range(counts_1[0]):
                keypoints_1 = get_keypoint(objects_1, i, peaks_1)
                both_keypoints.append(keypoints_1)
                frame_l = draw_keypoints(frame_l, keypoints_1)
            for i in range(counts_2[0]):
                keypoints_2 = get_keypoint(objects_2, i, peaks_2)
                both_keypoints.append(keypoints_2)
                frame_r= draw_keypoints(frame_r, keypoints_2)
            
            video_left.write(np.array(frame_l))
            video_right.write(np.array(frame_r))


            num_of_points = len(both_keypoints[0]) 
            for i in range(num_of_points):
                y_right  = both_keypoints[1][i][1]
                y_left = both_keypoints[0][i][1]
                if y_right is None or y_left is None:
                    npthing = 0
                else:
                    failure = (y_left - y_right)
                    if failure < 0:
                        failure = failure * -1
                    ave = (y_left + y_right)/2
                    failure = (failure / ave) * 100
                    if failure > 2:
                        fails+=1
            frames+=1



            coordinates = []
            if(save_data):
                if (counts_1[0] == 1) and (counts_2[0] == 1):
                    # print(both_keypoints)
                    coordinates = create_coordanites(both_keypoints)
                file = open("/home/thomas/Desktop/test.txt", 'a')
                for coord in coordinates:
                    for point in coord:
                        file.write(str(point))
                        file.write(" ")
                    file.write("\n")
                file.write("\n")
                file.write("next frame\n")
                file.close()
            print('posed frames')
            # camera_images = np.hstack((frame_l, frame_r))
            # camera_images =  cv2.resize(camera_images, (1640,616), interpolation = cv2.INTER_AREA)
            # cv2.imshow("CSI Cameras", camera_images)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            print("total frames: " + str(frames) + " total fails: " + str(fails))
            print('no frames')
            break
    left_capture.release()
    right_capture.release()
    video_left.release()
    video_right.release()
    cv2.destroyAllWindows()



def create_coordanites(two_views):
    x_length = 3.674 # (mm) sensor size
    y_length = 2.76 # (mm)
    focal_length = 3.15  # (mm)
    # x_length = 3280 # (pixel) sensor size
    # y_length = 2464 # (pixel)
    # focal_length = 2812.193  # (pixel)
    baseline = 164.13 # distance between the two cameras on the x-axis (mm)

    coordinates = []
    num_of_points = len(two_views[0]) 
    for i in range(num_of_points):
        # x_left  = two_views[1][i][1]
        # x_right = two_views[0][i][1]
        # y_left  = two_views[1][i][2]
        # y_right = two_views[0][i][2]
        y_right  = two_views[1][i][1]
        y_left = two_views[0][i][1]
        x_right  = two_views[1][i][2]
        x_left = two_views[0][i][2]

        #print('y left: ' + str(y_left) + ' y right: ' + str(y_right))
        if x_left is None or x_right is None or y_left is None or y_right is None:
            x = 0
            y = 0
            z = 0
        else:
            x_left  = x_left * x_length
            x_right = x_right * x_length
            y_left  = y_left * y_length
            y_right = y_right * y_length
            disparity = x_left - x_right
            z = (focal_length * baseline)/disparity
            x = x_left * (z / focal_length)
            y = y_left * (z / focal_length)
        coordinates.append((x,y,z))    
    return coordinates





create_pose_video('test.avi', True)
# video_recorder.playback_video('pose_test.avi')