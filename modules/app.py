import threading
import time

import cv2

from modules import capturer
from modules.compositor import Compositor
from modules.face_detector import FaceDetector
from modules.face_swapper import FaceSwapper


class FaceSwapApp:
    def __init__(self):
        self.models_loaded = False
        self.is_running = False
        self.frame_lock = threading.Lock()
        self.current_frame = None
        self.source_face = None

        # Settings state
        self.settings = {"detect": True, "swap": True, "blend": True}

        # Khởi tạo models trong thread riêng để không đơ UI lúc mở app
        threading.Thread(target=self._load_models, daemon=True).start()

    def _load_models(self):
        print("--- System: Loading AI Models... ---")
        try:
            self.face_detector = FaceDetector()
            self.face_swapper = FaceSwapper("models/inswapper_128.onnx")
            self.compositor = Compositor()
            self.models_loaded = True
            print("--- System: Models Ready! ---")
        except Exception as e:
            print(f"Error loading models: {e}")

    def set_source_image(self, path):
        """Xử lý logic load ảnh và detect khuôn mặt source"""
        if not self.models_loaded:
            return False, "Models are still loading...", None

        img = cv2.imread(path)
        if img is None:
            return False, "Cannot read image file", None

        faces = self.face_detector.detect(img)
        if not faces:
            return False, "No face detected in source image", None

        self.source_face = faces[0]
        return True, "Success", img

    def update_setting(self, key, value):
        if key in self.settings:
            self.settings[key] = value

    def start_camera(self):
        if not self.models_loaded:
            return False, "System initializing..."
        if self.source_face is None:
            return False, "Please upload a source image first"

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

        # Xóa frame cuối để màn hình đen
        with self.frame_lock:
            self.current_frame = None

    def get_frame(self):
        """UI gọi hàm này để lấy ảnh hiển thị"""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None

    def _process_loop(self):
        """Vòng lặp xử lý logic ngầm"""
        while self.is_running:
            frame = self.cap.read()
            if frame is None:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)  # Mirror effect

            # Logic xử lý AI
            if self.source_face and self.settings["detect"]:
                try:
                    faces = self.face_detector.detect(frame)
                    if faces:
                        target_face = faces[0]
                        if self.settings["swap"]:
                            orig_frame = frame.copy()  # giữ frame gốc cho mask/occlusion
                            swapped = self.face_swapper.swap(
                                frame, target_face, self.source_face
                            )

                            if self.settings["blend"]:
                                swapped = self.compositor.blend_composite(
                                    orig_frame,          # frame gốc để tính mask/occlusion
                                    swapped,             # frame đã swap
                                    target_face,
                                    parsing_mask=None,   # nếu chưa có parsing
                                    blur_amount=0.5,
                                )

                            # Enhance khuôn mặt (GFPGAN); nếu model không có sẽ trả về input
                            frame = self.compositor.enhance(swapped)
                except Exception as e:
                    print(f"Processing error: {e}")

            with self.frame_lock:
                self.current_frame = frame
