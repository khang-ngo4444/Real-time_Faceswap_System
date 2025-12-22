import modules.capturer as capturer
from modules.face_detector import FaceDetector
from modules.face_swapper import FaceSwapper
import cv2
import numpy as np


def mask_mouth(frame, swapped_frame, face):
    h,w, _ = frame.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    landmarks = face.landmark_2d_106.astype(np.int32)
    lower_lip_order = [
        64,
        63,
        67,
        68,
        69,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        0,
        8,
        7,
        6,
        5,
        4,
        3,
        2,
        65,
    ]
    lower_lip_landmarks = landmarks[lower_lip_order].astype(
        np.int32
    )
    hull = cv2.convexHull(lower_lip_landmarks)
    cv2.fillConvexPoly(mask, hull, 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask = cv2.dilate(mask, kernel, 1)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0

    final_frame = (
            frame * mask_3ch + swapped_frame * (1 - mask_3ch)
    ).astype(np.uint8)

    return final_frame


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
            mouth = mask_mouth(frame, swapped_frame, target_face)

            cv2.imshow("Mouth", mouth)
            # cv2.imshow("Swapped Face Raw Output", swapped_frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
