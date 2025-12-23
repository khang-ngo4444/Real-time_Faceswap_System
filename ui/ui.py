import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
from PIL import Image, ImageTk

import modules.capturer as capturer
from modules.compositor import Compositor
from modules.face_detector import FaceDetector
from modules.face_swapper import FaceSwapper


class FaceSwap:
    """Class chứa logic xử lý deepfake"""

    def __init__(self):
        self.cap = None
        self.face_detector = None
        self.face_swapper = None
        self.compositor = None
        self.source_face = None
        self.current_frame = None
        self.is_running = False
        self.models_initialized = False
        self.frame_lock = threading.Lock()

    def initialize_models(self):
        try:
            self.face_detector = FaceDetector()
            self.face_swapper = FaceSwapper("models/inswapper_128.onnx")
            self.compositor = Compositor()
            self.models_initialized = True
            return True, None
        except Exception as e:
            return False, str(e)

    def load_source_image(self, image_path):
        try:
            source_img = cv2.imread(image_path)
            if source_img is None:
                return False, "Cannot read image file"

            faces = self.face_detector.detect(img=source_img)
            if not faces:
                return False, "No face detected in the image"

            self.source_face = faces[0]
            return True, None
        except Exception as e:
            return False, str(e)

    def start_camera(self):
        if not self.models_initialized:
            return False, "Models not initialized"
        if self.source_face is None:
            return False, "No source face loaded"

        try:
            self.cap = capturer.Camera()
            self.is_running = True
            threading.Thread(target=self._process_loop, daemon=True).start()
            return True, None
        except Exception as e:
            return False, str(e)

    def stop_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        with self.frame_lock:
            self.current_frame = None

    def get_current_frame(self):
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def _process_loop(self):
        while self.is_running:
            try:
                frame = self.cap.read()
                if frame is None:
                    continue

                display_frame = frame.copy()

                if self.source_face is not None:
                    faces = self.face_detector.detect(frame)
                    if faces:
                        target_face = faces[0]
                        if target_face.landmark_2d_106 is not None:
                            swapped_frame = self.face_swapper.swap(
                                frame, target_face, self.source_face
                            )
                            display_frame = self.compositor.blend_mouth_mask(
                                frame, swapped_frame, target_face
                            )

                with self.frame_lock:
                    self.current_frame = display_frame
            except Exception as e:
                print(f"Error in processing loop: {e}")
                continue

    def cleanup(self):
        if self.is_running:
            self.stop_camera()


# Hàm run ở NGOÀI class
def run(deepfake_app):
    """Chạy giao diện GUI"""

    root = tk.Tk()
    root.title("Face Swap App")
    root.geometry("900x700")

    video_label = tk.Label(root, bg="black")
    video_label.pack(pady=10)

    status_label = tk.Label(root, text="Status: Not initialized", font=("Arial", 10))
    status_label.pack(pady=5)

    def update_status(text):
        status_label.config(text=f"Status: {text}")

    def btn_initialize():
        update_status("Initializing models...")
        root.update()
        success, error = deepfake_app.initialize_models()
        if success:
            update_status("Models initialized ✓")
            messagebox.showinfo("Success", "Models ready!")
        else:
            update_status("Failed to initialize")
            messagebox.showerror("Error", error)

    def btn_load_image():
        file_path = filedialog.askopenfilename(
            title="Select source image", filetypes=[("Images", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            success, error = deepfake_app.load_source_image(file_path)
            if success:
                update_status("Source image loaded ✓")
            else:
                messagebox.showerror("Error", error)

    def btn_start():
        success, error = deepfake_app.start_camera()
        if success:
            update_status("Camera running...")
            update_video()
        else:
            messagebox.showerror("Error", error)

    def btn_stop():
        deepfake_app.stop_camera()
        update_status("Camera stopped")
        video_label.config(image="")

    def update_video():
        if deepfake_app.is_running:
            frame = deepfake_app.get_current_frame()
            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((640, 480))
                imgtk = ImageTk.PhotoImage(image=img)
                video_label.imgtk = imgtk
                video_label.config(image=imgtk)

            root.after(30, update_video)

    def on_close():
        deepfake_app.cleanup()
        root.destroy()

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=10)

    tk.Button(
        btn_frame, text="1. Initialize Models", command=btn_initialize, width=20
    ).pack(side=tk.LEFT, padx=5)
    tk.Button(
        btn_frame, text="2. Load Source Image", command=btn_load_image, width=20
    ).pack(side=tk.LEFT, padx=5)
    tk.Button(btn_frame, text="3. Start Camera", command=btn_start, width=20).pack(
        side=tk.LEFT, padx=5
    )
    tk.Button(
        btn_frame, text="Stop", command=btn_stop, width=10, bg="red", fg="white"
    ).pack(side=tk.LEFT, padx=5)

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
