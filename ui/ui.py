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

        # Feature toggles
        self.face_detection_enabled = True
        self.face_swap_enabled = True
        self.mouth_blend_enabled = True

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

    def toggle_face_detection(self):
        self.face_detection_enabled = not self.face_detection_enabled
        return self.face_detection_enabled

    def toggle_face_swap(self):
        self.face_swap_enabled = not self.face_swap_enabled
        return self.face_swap_enabled

    def toggle_mouth_blend(self):
        self.mouth_blend_enabled = not self.mouth_blend_enabled
        return self.mouth_blend_enabled

    def _process_loop(self):
        while self.is_running:
            try:
                frame = self.cap.read()
                if frame is None:
                    continue

                display_frame = frame.copy()

                # Chỉ xử lý nếu có source face và face detection được bật
                if self.source_face is not None and self.face_detection_enabled:
                    faces = self.face_detector.detect(frame)

                    if faces:
                        target_face = faces[0]

                        if target_face.landmark_2d_106 is not None:
                            # Face swap nếu được bật
                            if self.face_swap_enabled:
                                swapped_frame = self.face_swapper.swap(
                                    frame, target_face, self.source_face
                                )

                                # Mouth blend nếu được bật
                                if self.mouth_blend_enabled:
                                    display_frame = self.compositor.blend_mouth_mask(
                                        frame, swapped_frame, target_face
                                    )
                                else:
                                    display_frame = swapped_frame

                with self.frame_lock:
                    self.current_frame = display_frame
            except Exception as e:
                print(f"Error in processing loop: {e}")
                continue

    def cleanup(self):
        if self.is_running:
            self.stop_camera()


def run(face_swap_app):
    """Chạy giao diện GUI"""

    root = tk.Tk()
    root.title("Face Swap App - Advanced Controls")
    root.geometry("950x850")  # Tăng chiều cao một chút để chứa các nút

    # --- SỬA LỖI MÀN HÌNH ĐEN TẠI ĐÂY ---
    # Tạo Frame chứa video với kích thước pixel cố định
    video_container = tk.Frame(root, width=640, height=480, bg="black")
    video_container.pack(pady=10)
    video_container.pack_propagate(False)  # Ngăn Frame co lại theo Label bên trong

    # Label hiển thị video nằm bên trong Frame
    video_label = tk.Label(video_container, bg="black")
    video_label.pack(fill=tk.BOTH, expand=True)
    # ------------------------------------

    # Status
    status_label = tk.Label(root, text="Status: Not initialized", font=("Arial", 11))
    status_label.pack(pady=5)

    def update_status(text):
        status_label.config(text=f"Status: {text}")

    # Main controls frame
    main_controls = tk.Frame(root)
    main_controls.pack(pady=10)

    def btn_initialize():
        update_status("Initializing models...")
        root.update()
        success, error = face_swap_app.initialize_models()
        if success:
            update_status("Models initialized ✓")
            messagebox.showinfo("Success", "Models ready!")
            enable_feature_toggles()
        else:
            update_status("Failed to initialize")
            messagebox.showerror("Error", error)

    def btn_load_image():
        file_path = filedialog.askopenfilename(
            title="Select source image", filetypes=[("Images", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            success, error = face_swap_app.load_source_image(file_path)
            if success:
                update_status("Source image loaded ✓")
                messagebox.showinfo("Success", "Source image loaded!")
            else:
                messagebox.showerror("Error", error)

    def btn_start():
        success, error = face_swap_app.start_camera()
        if success:
            update_status("Camera running...")
            update_video()
        else:
            messagebox.showerror("Error", error)

    def btn_stop():
        face_swap_app.stop_camera()
        update_status("Camera stopped")
        video_label.config(image="")

    def update_video():
        if face_swap_app.is_running:
            frame = face_swap_app.get_current_frame()
            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((640, 480))
                imgtk = ImageTk.PhotoImage(image=img)
                video_label.imgtk = imgtk
                video_label.config(image=imgtk)

            root.after(30, update_video)

    def on_close():
        face_swap_app.cleanup()
        root.destroy()

    # Main control buttons
    tk.Button(
        main_controls,
        text="1. Initialize Models",
        command=btn_initialize,
        width=18,
        height=2,
    ).pack(side=tk.LEFT, padx=5)

    tk.Button(
        main_controls,
        text="2. Load Source Image",
        command=btn_load_image,
        width=18,
        height=2,
    ).pack(side=tk.LEFT, padx=5)

    tk.Button(
        main_controls,
        text="3. Start Camera",
        command=btn_start,
        width=18,
        height=2,
        bg="#28a745",
        fg="white",
    ).pack(side=tk.LEFT, padx=5)

    tk.Button(
        main_controls,
        text="Stop",
        command=btn_stop,
        width=12,
        height=2,
        bg="#dc3545",
        fg="white",
    ).pack(side=tk.LEFT, padx=5)

    # Feature toggles frame
    toggle_frame = tk.LabelFrame(root, text="Feature Controls", padx=10, pady=10)
    toggle_frame.pack(pady=10, padx=20, fill=tk.X)

    # Toggle buttons
    face_detect_btn = tk.Button(
        toggle_frame,
        text="Face Detection: ON",
        width=25,
        height=2,
        bg="#28a745",
        fg="white",
        state=tk.DISABLED,
    )
    face_detect_btn.pack(side=tk.LEFT, padx=10)

    face_swap_btn = tk.Button(
        toggle_frame,
        text="Face Swap: ON",
        width=25,
        height=2,
        bg="#28a745",
        fg="white",
        state=tk.DISABLED,
    )
    face_swap_btn.pack(side=tk.LEFT, padx=10)

    mouth_blend_btn = tk.Button(
        toggle_frame,
        text="Mouth Blend: ON",
        width=25,
        height=2,
        bg="#28a745",
        fg="white",
        state=tk.DISABLED,
    )
    mouth_blend_btn.pack(side=tk.LEFT, padx=10)

    def toggle_face_detection():
        enabled = face_swap_app.toggle_face_detection()
        if enabled:
            face_detect_btn.config(text="Face Detection: ON", bg="#28a745")
        else:
            face_detect_btn.config(text="Face Detection: OFF", bg="#6c757d")

    def toggle_face_swap():
        enabled = face_swap_app.toggle_face_swap()
        if enabled:
            face_swap_btn.config(text="Face Swap: ON", bg="#28a745")
        else:
            face_swap_btn.config(text="Face Swap: OFF", bg="#6c757d")

    def toggle_mouth_blend():
        enabled = face_swap_app.toggle_mouth_blend()
        if enabled:
            mouth_blend_btn.config(text="Mouth Blend: ON", bg="#28a745")
        else:
            mouth_blend_btn.config(text="Mouth Blend: OFF", bg="#6c757d")

    def enable_feature_toggles():
        face_detect_btn.config(state=tk.NORMAL, command=toggle_face_detection)
        face_swap_btn.config(state=tk.NORMAL, command=toggle_face_swap)
        mouth_blend_btn.config(state=tk.NORMAL, command=toggle_mouth_blend)

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
