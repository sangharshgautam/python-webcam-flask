from PIL import Image

class Detect_object(object):

    def __init__(self):
        pass

    def detect(self, img):
        return img.transpose(Image.FLIP_LEFT_RIGHT)
