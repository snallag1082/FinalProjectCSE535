import os
import shutil
from pathlib import Path
from statistics import mode
from pandas import DataFrame
from sklearn.metrics import classification_report
from alphabet_mode_main import predict_labels_from_frames
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 



PATH_TO_VIDEOS = './Letters/Videos'
PATH_TO_FRAMES = './Letters/Frames'
PATH_TO_HAND_FRAMES = './Letters/Hand_Frames'
PATH_TO_COMBINED_HAND_FRAMES = './Letters/Combined_Hand_Frames'
PATH_TO_RESULTS = './Letters/results.csv'


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
    for root, dirs, files in os.walk(PATH_TO_COMBINED_HAND_FRAMES):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))


def generatekeypoints_posenet():
    print('\n==========Generating Posenet Keypoints for Alphabets==========\n')
    for root, dirs, files in os.walk(PATH_TO_VIDEOS):
        for dir in dirs:
            path_to_videos, path_to_frames = os.path.join(PATH_TO_VIDEOS, dir), os.path.join(PATH_TO_FRAMES, dir)
            if not os.path.exists(path_to_frames):
                os.mkdir(path_to_frames)
            print('\nPerson: ' + dir)
            os.system('python ./posenet/Frames_Extractor.py --path_to_videos=%s --path_to_frames=%s' % (path_to_videos, path_to_frames))
            os.system('node ./posenet/scale_to_videos.js %s' % (path_to_frames))
            os.system('python ./posenet/convert_to_csv.py --path_to_videos=%s --path_to_frames=%s' % (path_to_videos, path_to_frames))


def handframes_extraction():
    for root, dirs, files in os.walk(PATH_TO_VIDEOS):
        for video in files:
            path_to_file = os.path.join(root, video)
            path_to_frames, path_to_hand_frames = os.path.join(PATH_TO_FRAMES, Path(path_to_file).parent.name), os.path.join(PATH_TO_HAND_FRAMES, Path(path_to_file).parent.name)
            nameofvideo = video.split('.')[0]
            print('\n========== Extracting Hand Frames for Alphabet: {0} ==========\n'.format(path_to_file))

            if not os.path.exists(path_to_hand_frames):
                os.mkdir(path_to_hand_frames)
            
            os.system('python ./hand_extractor.py --path_to_frames=%s --path_to_hand_frames=%s' % (path_to_frames + '/' + nameofvideo, path_to_hand_frames + '/' + nameofvideo))
            print('Extracting Hand Frames for Alphabet: {0}'.format(path_to_file))


def combinedframes_generator():
    print('\n==========Combining Hand Frames of each Alphabet==========\n')
    alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    for alphabet in alphabets:
        count = 0
        if not os.path.exists(PATH_TO_COMBINED_HAND_FRAMES + '/' + alphabet):
            os.makedirs(PATH_TO_COMBINED_HAND_FRAMES + '/' + alphabet)
        for root, dirs, files in os.walk(PATH_TO_VIDEOS):
            for dir in dirs:
                for r, d, fls in os.walk(os.path.join(PATH_TO_HAND_FRAMES, dir, alphabet)):
                    for f in fls:
                        shutil.copy2(os.path.join(r, f), PATH_TO_COMBINED_HAND_FRAMES + '/' + alphabet + '/' + str(count) + '.png')
                        count += 1


def alphabets_predictor(): 
    output = []
    for root, dirs, files in os.walk(PATH_TO_COMBINED_HAND_FRAMES):
        for dir in dirs:
            print('\n========== Predict Alphabet: {0} ==========\n'.format(dir))
            pred_frames = predict_labels_from_frames(PATH_TO_COMBINED_HAND_FRAMES + '/' + dir)
            prediction = mode(pred_frames)
            print('Prediction: {0}\tGround Truth: {1}'.format(prediction, dir))
            output.append([prediction, dir])

    dframe = DataFrame(output, columns=['predicted', 'ground_truth'])
    print(classification_report(dframe.predicted, dframe.ground_truth))
    dframe.to_csv(PATH_TO_RESULTS)



clean_directories()

# Folder to save hand frames
if not os.path.exists(PATH_TO_FRAMES):
    os.makedirs(PATH_TO_FRAMES)
if not os.path.exists(PATH_TO_HAND_FRAMES):
    os.makedirs(PATH_TO_HAND_FRAMES)
if not os.path.exists(PATH_TO_COMBINED_HAND_FRAMES):
    os.makedirs(PATH_TO_COMBINED_HAND_FRAMES)

# Create posenet wrist points
generatekeypoints_posenet()

# extract hand frames from videos
handframes_extraction()

# combined hand frames of each letter into one folder
combinedframes_generator()

# train model on kaggle dataset
os.system('python ./cnn_model.py --path_to_dataset=%s --save_model=%s' % ('./asl-alphabet/asl_alphabet_train/', 'cnn_model.h5'))

#predict alphabet
alphabets_predictor()

# finetune model on our dataset
os.system('python ./cnn_model.py --path_to_dataset=%s --save_model=%s --load_model=%s' % (PATH_TO_COMBINED_HAND_FRAMES + '/', 'cnn_model.h5', 'cnn_model.h5'))

#predict alphabet
alphabets_predictor()