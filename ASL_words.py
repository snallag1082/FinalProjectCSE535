import os
import math
import shutil
import pandas as pd
from pandas import DataFrame
from statistics import mode
from alphabet_mode_main import predict_words_from_frames
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

PATH_TO_VIDEOS = './Words/Videos'
PATH_TO_FRAMES = './Words/Frames'
PATH_TO_HAND_FRAMES = './Words/Hand_Frames'
PATH_TO_RESULTS = './Words/results.csv'


def clean_directories():
    for root, dirs, files in os.walk(PATH_TO_FRAMES):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))

    for root, dirs, files in os.walk(PATH_TO_HAND_FRAMES):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))


def generatekeypoints_posenet():
    print('\n==========Generating Posenet Keypoints for Videos==========\n')
    os.system('python ./posenet/Frames_Extractor.py --path_to_videos=%s --path_to_frames=%s' % (PATH_TO_VIDEOS, PATH_TO_FRAMES))
    os.system('node ./posenet/scale_to_videos.js %s' % (PATH_TO_FRAMES))
    os.system('python ./posenet/convert_to_csv.py --path_to_videos=%s --path_to_frames=%s' % (PATH_TO_VIDEOS, PATH_TO_FRAMES))


def segment_videos(video_name):
    # We'll calculate the difference between positions of rights hand wrist points in consecutive frames. 
    # If the difference is greater than a certain threshold, we'll assume that there is a transition of letters of the word.
    print('\n========== Segmenting Video: {0} ==========\n'.format(video_name))
    keyptPosenet = pd.read_csv(PATH_TO_FRAMES + '/' + video_name + '/' + 'key_points.csv')
    threshold = 20
    rightWrist_x = keyptPosenet.rightWrist_x
    rightWrist_y = keyptPosenet.rightWrist_y
    arr_frame = []

    for i in range(keyptPosenet.shape[0]-1):
        dist = math.sqrt( ((rightWrist_x[i + 1] - rightWrist_x[i]) ** 2) + ((rightWrist_y[i + 1] - rightWrist_y[i]) ** 2))
        if dist < threshold and rightWrist_y[i] < 600:
            arr_frame.append(i)
    
    frames = []
    start = 0
    end = start

    for i in range(1, len(arr_frame)):
        diff = arr_frame[i] - arr_frame[i-1]
        if diff == 1 or diff == 2 or diff == 3:
            end = arr_frame[i]
            continue

        if (end - start) >= 50:
            frames.append([start, end])
        start = arr_frame[i]
        end = start
    
    if (end - start) >= 50:
            frames.append([start, end])

    print('Frames: ', frames)

    return frames


def predict_word(frames, video_name): 
    print('\n========== Predict Video: {0} ==========\n'.format(video_name))
    letters = []
    for i in range(len(frames)):
        frames_pred = predict_words_from_frames(PATH_TO_HAND_FRAMES + '/' + video_name, frames[i][0], frames[i][1])
        prediction = mode(frames_pred)
        letters.append(prediction)

    return ''.join(letters).upper()


def classification_report(df):
    accuracy = []
    for index, row in df.iterrows():
        count = 0
        for i in range(min(len(row['ground_truth']), len(row['predicted']))):
            if row['ground_truth'][i] == row['predicted'][i]:
                count += 1
        accuracy.append(count * 100 / len(row['ground_truth']))
    df['accuracy (%)'] = accuracy
    return df



clean_directories()

# Folder to save hand frames
if not os.path.exists(PATH_TO_FRAMES):
    os.makedirs(PATH_TO_FRAMES)
if not os.path.exists(PATH_TO_HAND_FRAMES):
    os.makedirs(PATH_TO_HAND_FRAMES)

# Initialise predicted array
output = []

# Create posenet wrist points
generatekeypoints_posenet()

for root, dirs, files in os.walk(PATH_TO_VIDEOS):
    for video in files:

        print('\n' + '-'*100)

        video_name = video.split('.')[0]

        print('\n========== Extracting Hand Frames for Video: {0} ==========\n'.format(video))
        os.system('python ./hand_extractor.py --path_to_frames=%s --path_to_hand_frames=%s' % (PATH_TO_FRAMES + '/' + video_name, PATH_TO_HAND_FRAMES + '/' + video_name))
        print('Hand Frames extracted for Video: {0}'.format(video))

        frames = segment_videos(video_name)

        prediction = predict_word(frames, video_name)
        print('Prediction: {0}\tGround Truth: {1}'.format(prediction, video_name))
        output.append([prediction, video_name])


df = DataFrame(output, columns=['predicted', 'ground_truth'])
print('\n' + '-'*100)
df = classification_report(df)
print(df)
print('\nAverage Accuracy: {0}%\n'.format(df['accuracy (%)'].mean()))
df.to_csv(PATH_TO_RESULTS)
