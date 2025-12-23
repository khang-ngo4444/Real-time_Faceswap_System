import modules.capturer as capturer
from modules.compositor import Compositor
from modules.face_detector import FaceDetector
from modules.face_swapper import FaceSwapper
from modules.face_parser import FaceParser
import cv2

def run():
    cap = capturer.Camera()
    face_detector = FaceDetector()
    face_swapper = FaceSwapper("models/inswapper_128.onnx")
    face_parser = FaceParser()
    compositor = Compositor()

    source_img = cv2.imread("assets/sample_2.jpg")
    source_face = face_detector.detect(img=source_img)[0]

    while True:
        frame = cap.read()
        if frame is None: break
        
        faces = face_detector.detect(frame)

        if faces:
            target_face = faces[0]
            swapped_frame = face_swapper.swap(frame, target_face, source_face)
            parsing_mask = face_parser.parse(frame, target_face)
            
            final_frame = compositor.blend_composite(
                frame, 
                swapped_frame, 
                target_face, 
                parsing_mask,
                blur_amount=0.5  # 0.3-0.6 tùy chỉnh
            )

            cv2.imshow("Face Swap (Real Mouth + Hand Aware)", final_frame)
        else:
            cv2.imshow("Face Swap (Real Mouth + Hand Aware)", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()