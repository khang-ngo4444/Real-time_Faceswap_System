import modules.capturer as capturer
from modules.face_detector import FaceDetector
from modules.face_swapper import FaceSwapper

import cv2

def run():
    cap = capturer.Camera()
    face_detector = FaceDetector()
    face_swapper = FaceSwapper("models/inswapper_128.onnx")

    source_img = cv2.imread("assets/23020016.jpg")
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
            cv2.imshow("Swapped Face Raw Output", swapped_frame)

            # for x, y in target_face.landmark_2d_106.astype(int):
            #     cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            #
            # # Bounding box
            # x1, y1, x2, y2 = target_face.bbox.astype(int)
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
