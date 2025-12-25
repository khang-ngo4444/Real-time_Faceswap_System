import threading
import time

import cv2

# Import các modules
import modules.capturer as capturer
from modules.compositor import Compositor
from modules.face_detector import FaceDetector
from modules.face_swapper import FaceSwapper


class FaceSwapApp:
    def __init__(self):
        self.models_initialized = False
        self.is_running = False
        self.frame_lock = threading.Lock()
        self.current_frame = None
        self.source_face = None

        # Biến FPS
        self.fps = 0
        self.frame_count = 0
        self.start_time = 0

        # mouth_mask = True  => preserve mouth from original (use compositor masks)
        # mouth_mask = False => full-face swap (show swapped)
        self.settings = {"bounding_box": False, "swap": True, "mouth_mask": True, "enhance": False}

        threading.Thread(target=self._load_models, daemon=True).start()

    def _load_models(self):
        print("--- System: Loading AI Models... ---")
        try:
            self.face_detector = FaceDetector()
            self.face_swapper = FaceSwapper("models/inswapper_128.onnx")
            self.compositor = Compositor()
            self.models_initialized = True
            print("--- System: Models Ready! ---")
        except Exception as e:
            print(f"Error loading models: {e}")

    def set_source_image(self, path):
        if not self.models_initialized:
            return False, "Models are initializing", None
        img = cv2.imread(path)
        if img is None:
            return False, "Unable to upload image", None

        faces = self.face_detector.detect(img)
        if not faces:
            return False, "No face found from image", None

        self.source_face = faces[0]
        return True, "Success", img

    def update_setting(self, key, value):
        if key in self.settings:
            self.settings[key] = value

    def start_camera(self):
        if not self.models_initialized:
            return False, "Models are initializing"
        if self.source_face is None:
            return False, "No source image found"

        try:
            self.cap = capturer.Camera()
            self.is_running = True
            threading.Thread(target=self._process_loop, daemon=True).start()
            return True, None
        except Exception as e:
            return False, str(e)

    def stop_camera(self):
        self.is_running = False
        if hasattr(self, "cap") and self.cap:
            self.cap.release()
        with self.frame_lock:
            self.current_frame = None
            self.fps = 0  # Reset FPS

    def get_frame(self):
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    # Hàm mới để UI lấy FPS
    def get_fps(self):
        return self.fps

    def _process_loop(self):
        # Biến dùng để tính FPS trung bình mượt hơn
        prev_frame_time = 0

        while self.is_running:
            frame = self.cap.read()
            if frame is None:
                time.sleep(0.01)
                continue

            # Bắt đầu đo thời gian xử lý frame
            new_frame_time = time.time()

            frame = cv2.flip(frame, 1)

            if self.source_face:
                try:
                    faces = self.face_detector.detect(frame)
                    if faces:
                        target_face = faces[0]

                        if self.settings["swap"]:
                            orig_frame = frame.copy()  # giữ frame gốc cho mask/occlusion
                            swapped = self.face_swapper.swap(
                                frame, target_face, self.source_face
                            )
                            # Mouth-mask behavior:
                            # - mouth_mask True  => preserve mouth from original (use compositor.blend_composite)
                            # - mouth_mask False => full-face swap (show swapped)
                            if self.settings.get("mouth_mask", True):
                                try:
                                    frame = self.compositor.blend_composite(
                                        orig_frame,
                                        swapped,
                                        target_face,
                                        parsing_mask=None,
                                        blur_amount=0.5,
                                    )
                                except Exception as e:
                                    print(f"Blend Error: {e}")
                                    frame = swapped
                            else:
                                frame = swapped

                            if self.settings["enhance"]:
                                try:
                                    frame = self.compositor.enhance(frame)
                                except Exception as e:
                                    print(f"Enhance Error: {e}")

                                    swapped = self.compositor.blend_composite(
                                        orig_frame,          # frame gốc để tính mask/occlusion
                                        swapped,             # frame đã swap
                                        target_face,
                                        parsing_mask=None,   # nếu chưa có parsing
                                        blur_amount=0.5,
                                    )

                                # Enhance khuôn mặt (GFPGAN); nếu model không có sẽ trả về input
                                frame = self.compositor.enhance(swapped)

                            if self.settings["bounding_box"]:
                                face = self.face_detector.detect(frame)[0]
                                x1, y1, x2, y2 = face.bbox.astype(int)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 1)

                except Exception as e:
                    print(f"Processing Error: {e}")

            # Tính toán FPS
            fps_val = (
                1 / (new_frame_time - prev_frame_time)
                if (new_frame_time - prev_frame_time) > 0
                else 0
            )
            prev_frame_time = new_frame_time

            # Cập nhật biến FPS chung (làm tròn 1 số thập phân)
            self.fps = round(fps_val, 1)

            with self.frame_lock:
                self.current_frame = frame
