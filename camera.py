import threading
import binascii
from time import sleep
from utils import base64_to_pil_image, pil_image_to_base64
import numpy as np
import cv2
import os

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

class Camera(object):
    def __init__(self, makeup_artist, detect_object):
        self.to_process = []
        self.to_output = []
        self.makeup_artist = makeup_artist
        self.detect_object = detect_object
        print("[INFO] loading model...")
        prototypes = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MobileNetSSD_deploy.prototxt.txt')
        models = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MobileNetSSD_deploy.caffemodel')
        self.net = cv2.dnn.readNetFromCaffe(prototypes, models)
        thread = threading.Thread(target=self.keep_processing, args=())
        thread.daemon = True
        thread.start()

    def process_one(self):
        if not self.to_process:
            return

        # input is an ascii string.
        input_str = self.to_process.pop(0)

        # convert it to a pil image
        input_img = base64_to_pil_image(input_str)

        ################## where the hard work is done ############
        # output_img is an PIL image
        output_img = self.makeup_artist.apply_makeup(input_img)
        self.detect_object.detect_object(input_img)

        # output_str is a base64 string in ascii
        output_str = pil_image_to_base64(output_img)

        # convert eh base64 string in ascii to base64 string in _bytes_
        self.to_output.append(binascii.a2b_base64(output_str))

    def keep_processing(self):
        while True:
            self.process_one()
            sleep(0.01)

    def enqueue_input(self, input):
        self.to_process.append(input)

    def get_frame(self):
        while not self.to_output:
            sleep(0.05)
        return self.to_output.pop(0)
