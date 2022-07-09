import cv2
import pandas as pd
import os
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_to_frames',
        dest='path_to_frames',
        default=0,
        help='give a frames folder path')
    parser.add_argument(
        '--path_to_hand_frames',
        dest='path_to_hand_frames',
        default=0,
        help='give a hand frames folder path')
    args = parser.parse_args()

    path_to_frames = args.path_to_frames
    path_to_hand_frames = args.path_to_hand_frames

    if not os.path.isdir(path_to_hand_frames):
        os.mkdir(path_to_hand_frames)

    # get frames from path_to_frames
    frames =  [f for f in os.listdir(path_to_frames) if f.endswith('.png')]
    sorted_frames = sorted(frames, key=lambda x: int(os.path.splitext(x)[0]))

    # posenet key points
    key_points = pd.read_csv(path_to_frames + '/' + 'key_points.csv')
    rightWrist_x = key_points.rightWrist_x
    rightWrist_y = key_points.rightWrist_y
    
    f = 0
    for frame in sorted_frames:
        try:
            if f < len(rightWrist_x):
                # cropping the hand part from the frame using posenet wrist points
                image = cv2.imread(path_to_frames + '/' + frame)
                hand_frame = image[
                    max(round(rightWrist_y[f])-300, 0):round(rightWrist_y[f])+100, 
                    max(round(rightWrist_x[f])-200, 0):round(rightWrist_x[f])+200
                ]
                # saving the cropped hand to an image file
                cv2.imwrite(path_to_hand_frames + '/' + str(f) + '.png', hand_frame)
                f = f + 1
        except:
            f = f + 1
