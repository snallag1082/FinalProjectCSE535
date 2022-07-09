import cv2
import numpy as np
import tensorflow as tf


keras = tf.keras
load_model = keras.models.load_model
Model = keras.models.Model

"""
This is a Singleton class which bears the ml model in memory
"""
import os.path
BASE = os.path.dirname(os.path.abspath(__file__))
# Need to change constant 80 based on the video dimensions
vid_dim = 80

class HandShapeFeatureExtractor:
    __single = None

    @staticmethod
    def get_instance():
        if HandShapeFeatureExtractor.__single is None:
            HandShapeFeatureExtractor()
        return HandShapeFeatureExtractor.__single

    def __init__(self):
        if HandShapeFeatureExtractor.__single is None:
            real_model = load_model( os.path.join( BASE, 'cnn_model.h5' ) )
            self.model = real_model
            # HandShapeFeatureExtractor.__single = self

        else:
            raise Exception("This Class bears the model, so it is made Singleton")

    # private method to preprocess the image
    @staticmethod
    def __pre_process_input_image(crop):
        try:
            img = cv2.resize(crop, (200, 200))
            imagearray = np.array(img) / 255.0
            imagearray = imagearray.reshape(1, 200, 200, 1)
            return imagearray
        except Exception as e:
            print(str(e))
            raise

    # calculating dimensions f0r the cropping the specific hand parts

    @staticmethod
    def __bound_box(x, y, max_y, max_x):
        y1 = y + vid_dim
        y2 = y - vid_dim
        x1 = x + vid_dim
        x2 = x - vid_dim
        if max_y < y1:
            y1 = max_y
        if y - vid_dim < 0:
            y2 = 0
        if x + vid_dim > max_x:
            x1 = max_x
        if x - vid_dim < 0:
            x2 = 0
        return y1, y2, x1, x2

    def extract_feature(self, image):
        try:
            imagetest = tf.keras.utils.load_img(image, target_size = (256, 256))
            imagetest = tf.keras.utils.img_to_array(imagetest)
            imagetest = np.expand_dims(imagetest, axis = 0)
            return self.model.predict(imagetest)
        except Exception as e:
            raise


