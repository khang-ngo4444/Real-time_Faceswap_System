import cv2

class Camera:
    """
    This class is mostly completed, maybe some OS related stuffs could be added
    """

    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)

    def read(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()