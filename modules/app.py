import modules.capturer as capturer
from modules.face_detector import FaceDetector
from modules.face_swapper import FaceSwapper
import cv2
import numpy as np


def mask_mouth(frame, face):
    for i, (x, y) in enumerate(face.landmark_2d_106.astype(int)):
        if 52 <= i <= 71:
            # Mouth landmarks
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        else:
            # Other face landmarks
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)        #

    return frame


def run():
    cap = capturer.Camera()
    face_detector = FaceDetector()
    face_swapper = FaceSwapper("models/inswapper_128.onnx")

    source_img = cv2.imread("assets/sample_5.jpg")
    source_face = face_detector.detect(img=source_img)[0]

    while True:
        frame = cap.read()
        faces = face_detector.detect(frame)

        if faces:
            target_face = faces[0] # only pick 1 face atm

            if target_face.landmark_2d_106 is None:
                print("Warning: landmark=None")
                continue

            # Face swap
            swapped_frame = face_swapper.swap(frame, target_face, source_face)
            mouth = mask_mouth(frame, target_face)

            cv2.imshow("Mouth", mouth)
            cv2.imshow("Swapped Face Raw Output", swapped_frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
