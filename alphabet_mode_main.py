import glob
import numpy as np
import os
import tensorflow as tf
from handshape_feature_extractor import HandShapeFeatureExtractor


def get_inference_vector_one_frame_alphabet(files_list):
    shape = HandShapeFeatureExtractor()
    retvect = []
    for vf in files_list:
        results = shape.extract_feature(vf)
        retvect.append(np.argmax(results))

    return retvect

def load_labels(label_file):
    label = []
    conv_ascii_lines = tf.io.gfile.GFile(label_file).readlines()
    for i in conv_ascii_lines:
        label.append(i.rstrip())
    return label

def load_label_dicts(label_file):
    id_to_labels = load_labels(label_file)
    labels_to_id = {}
    j = 0

    for id in id_to_labels:
        labels_to_id[id] = j
        j += 1

    return id_to_labels, labels_to_id


def predict_labels_from_frames(video_folder_path):
    conv_framestofiles = []
    path = os.path.join(video_folder_path, "*.png")
    frames = glob.glob(path)
    frames.sort()
    conv_framestofiles = frames
    
    pred_vector = get_inference_vector_one_frame_alphabet(conv_framestofiles)
    
    label_file = 'output_labels_alphabet.txt'
    id_to_labels, labels_to_id = load_label_dicts(label_file)
    
    retfinal_pred=[]
    
    for i in range(len(pred_vector)):
        for ins in labels_to_id:
            if pred_vector[i] == labels_to_id[ins]:
                retfinal_pred.append(ins)
    
    return retfinal_pred

def predict_words_from_frames(video_folder_path, start, end):
    files = []
    path = os.path.join(video_folder_path, "*.png")
    frames = glob.glob(path)
    nameofframe_arr = [video_folder_path + '\\' + str(i) + '.png' for i in range(start, end+1)]
    files = [frame for frame in frames if frame in nameofframe_arr]
    files.sort()
    pred_vector = get_inference_vector_one_frame_alphabet(files)
    label_file = 'output_labels_alphabet.txt'
    id_to_labels, labels_to_id = load_label_dicts(label_file)
    retfinal_pred=[]
    for i in range(len(pred_vector)):
        for ins in labels_to_id:
            if pred_vector[i] == labels_to_id[ins]:
                retfinal_pred.append(ins)
    
    return retfinal_pred