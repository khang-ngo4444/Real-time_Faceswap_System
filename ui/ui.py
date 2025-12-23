import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
from PIL import Image, ImageTk

# Import các modules xử lý (Giữ nguyên)
import modules.capturer as capturer
from modules.compositor import Compositor
from modules.face_detector import FaceDetector
from modules.face_swapper import FaceSwapper

# --- THEME COLORS ---
THEME = {
    "primary": "#2196F3",  # Blue
    "primary_dark": "#1976D2",
    "background": "#F0F2F5",  # Xám nhạt (Nền trái)
    "surface": "#FFFFFF",  # Trắng (Nền thẻ điều khiển)
    "video_bg": "#000000",  # Đen (Nền phải)
    "text_main": "#1A1A1A",
    "text_sub": "#757575",
    "border": "#E0E0E0",
}


class FaceSwap:
    """Class logic giữ nguyên từ file gốc"""

    def __init__(self):
        print("--- Loading Models... ---")
        try:
            self.face_detector = FaceDetector()
            self.face_swapper = FaceSwapper("models/inswapper_128.onnx")
            self.compositor = Compositor()
            self.models_initialized = True
            print("--- Models Loaded! ---")
        except Exception as e:
            print(f"Error loading models: {e}")
            self.models_initialized = False

        self.cap = None
        self.source_face = None
        self.current_frame = None
        self.is_running = False
        self.frame_lock = threading.Lock()

        self.face_detection_enabled = True
        self.face_swap_enabled = True
        self.mouth_blend_enabled = True

    def load_source_image(self, image_path):
        try:
            source_img = cv2.imread(image_path)
            if source_img is None:
                return False, "Cannot read image", None
            faces = self.face_detector.detect(img=source_img)
            if not faces:
                return False, "No face detected", None
            self.source_face = faces[0]
            return True, "Success", source_img
        except Exception as e:
            return False, str(e), None

    def start_camera(self):
        if not self.models_initialized:
            return False, "Models not loaded"
        if self.source_face is None:
            return False, "Select source face first"
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
                frame = cv2.flip(frame, 1)
                display_frame = frame.copy()

                if self.source_face and self.face_detection_enabled:
                    faces = self.face_detector.detect(frame)
                    if faces:
                        target_face = faces[0]
                        if target_face.landmark_2d_106 is not None:
                            if self.face_swap_enabled:
                                swapped_frame = self.face_swapper.swap(
                                    frame, target_face, self.source_face
                                )
                                if self.mouth_blend_enabled:
                                    display_frame = self.compositor.blend_mouth_mask(
                                        frame, swapped_frame, target_face
                                    )
                                else:
                                    display_frame = swapped_frame
                with self.frame_lock:
                    self.current_frame = display_frame
            except:
                continue

    def cleanup(self):
        self.stop_camera()


# --- CUSTOM WIDGETS ---


class RoundedButton(tk.Canvas):
    def __init__(
        self,
        parent,
        text,
        command=None,
        width=200,
        height=45,
        bg_color=THEME["primary"],
        hover_color=THEME["primary_dark"],
        text_color="white",
    ):
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=THEME["surface"],
            highlightthickness=0,
        )
        self.command = command
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.radius = height / 2

        self.rect = self._create_rounded_rect(
            0, 0, width, height, self.radius, fill=bg_color
        )
        self.text = self.create_text(
            width / 2,
            height / 2,
            text=text,
            fill=text_color,
            font=("Segoe UI", 10, "bold"),
        )

        self.bind(
            "<Enter>", lambda e: self.itemconfig(self.rect, fill=self.hover_color)
        )
        self.bind("<Leave>", lambda e: self.itemconfig(self.rect, fill=self.bg_color))
        self.bind("<Button-1>", lambda e: self.command() if self.command else None)
        self.configure(cursor="hand2")

    def _create_rounded_rect(self, x1, y1, x2, y2, r, **kwargs):
        points = [
            x1 + r,
            y1,
            x1 + r,
            y1,
            x2 - r,
            y1,
            x2 - r,
            y1,
            x2,
            y1,
            x2,
            y1 + r,
            x2,
            y1 + r,
            x2,
            y2 - r,
            x2,
            y2 - r,
            x2,
            y2,
            x2 - r,
            y2,
            x2 - r,
            y2,
            x1 + r,
            y2,
            x1 + r,
            y2,
            x1,
            y2,
            x1,
            y2 - r,
            x1,
            y2 - r,
            x1,
            y1 + r,
            x1,
            y1 + r,
            x1,
            y1,
        ]
        return self.create_polygon(points, smooth=True, **kwargs)


class MaterialSwitch(tk.Canvas):
    def __init__(self, parent, variable, command=None, bg=THEME["surface"]):
        super().__init__(
            parent, width=46, height=24, bg=bg, highlightthickness=0, cursor="hand2"
        )
        self.variable = variable
        self.command = command
        self.is_on = variable.get()
        self.bind("<Button-1>", self.toggle)
        self.render()

    def render(self):
        self.delete("all")
        track_col = "#90CAF9" if self.is_on else "#B0BEC5"
        thumb_col = THEME["primary"] if self.is_on else "#ECEFF1"
        # Draw Track
        self.create_polygon(
            [10, 4, 36, 4, 36, 20, 10, 20],
            smooth=True,
            width=16,
            outline=track_col,
            fill=track_col,
        )
        # Draw Thumb
        x = 34 if self.is_on else 12
        self.create_oval(x - 10, 2, x + 10, 22, fill=thumb_col, outline="")

    def toggle(self, event=None):
        self.is_on = not self.is_on
        self.variable.set(self.is_on)
        self.render()
        if self.command:
            self.command()


# --- MAIN UI ---


class FaceSwapUI:
    def __init__(self, face_swap_app):
        self.app = face_swap_app
        self.root = tk.Tk()
        self.setup_window()
        self.create_layout()

    def setup_window(self):
        self.root.title("Face Swap Studio")
        # Kích thước lớn để thấy rõ hiệu ứng tràn viền
        w, h = 1200, 720
        x = (self.root.winfo_screenwidth() - w) // 2
        y = (self.root.winfo_screenheight() - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        self.root.configure(bg=THEME["background"])

    def create_layout(self):
        # Container chính chia đôi màn hình: Trái (Controls) - Phải (Video Full)
        container = tk.Frame(self.root, bg=THEME["background"])
        container.pack(fill=tk.BOTH, expand=True)

        # 1. CỘT TRÁI (CONTROLS)
        # Width cố định, có padding bao quanh để tạo cảm giác "nổi"
        left_panel = tk.Frame(container, bg=THEME["background"], width=340)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=20, pady=20)
        left_panel.pack_propagate(False)

        self.create_control_card(left_panel)

        # 2. CỘT PHẢI (VIDEO FULL)
        # Không padx, không pady -> Tràn viền
        self.right_panel = tk.Frame(container, bg=THEME["video_bg"])
        self.right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Video Label lấp đầy cột phải
        self.video_label = tk.Label(
            self.right_panel,
            bg=THEME["video_bg"],
            text="Camera Off",
            fg="#333",
            font=("Segoe UI", 20),
        )
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Badge trạng thái (Overlay lên trên video)
        self.create_status_badge()

    def create_control_card(self, parent):
        # Thẻ trắng chứa controls
        card = tk.Frame(
            parent,
            bg=THEME["surface"],
            highlightbackground=THEME["border"],
            highlightthickness=1,
        )
        card.pack(fill=tk.BOTH, expand=True)

        # Nội dung bên trong thẻ
        content = tk.Frame(card, bg=THEME["surface"])
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Title
        tk.Label(
            content,
            text="Face Swap Studio",
            bg=THEME["surface"],
            fg=THEME["primary"],
            font=("Segoe UI", 16, "bold"),
        ).pack(anchor="w", pady=(0, 20))

        # --- Source Image ---
        tk.Label(
            content,
            text="Source Face",
            bg=THEME["surface"],
            fg=THEME["text_sub"],
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", pady=(0, 5))

        # Preview Box
        self.preview_frame = tk.Frame(content, bg=THEME["background"], height=180)
        self.preview_frame.pack(fill=tk.X, pady=(0, 10))
        self.preview_frame.pack_propagate(False)

        self.preview_label = tk.Label(
            self.preview_frame, text="No Image", bg=THEME["background"], fg="#999"
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        # Button Upload
        RoundedButton(
            content,
            text="Upload Image",
            command=self.load_image,
            bg_color="#E3F2FD",
            text_color=THEME["primary"],
            width=258,
            height=40,
        ).pack()

        tk.Frame(content, bg=THEME["border"], height=1).pack(fill=tk.X, pady=20)

        # --- Settings ---
        self.var_detect = tk.BooleanVar(value=True)
        self.var_swap = tk.BooleanVar(value=True)
        self.var_blend = tk.BooleanVar(value=True)

        self.add_toggle(content, "Face Detection", self.var_detect)
        self.add_toggle(content, "Face Swapping", self.var_swap)
        self.add_toggle(content, "Mouth Blending", self.var_blend)

        # Spacer để đẩy nút Start xuống đáy
        tk.Frame(content, bg=THEME["surface"]).pack(fill=tk.BOTH, expand=True)

        # --- Start/Stop Buttons ---
        self.btn_start = RoundedButton(
            content,
            text="START CAMERA",
            command=self.start_camera,
            width=258,
            height=50,
        )
        self.btn_start.pack(side=tk.BOTTOM)

        self.btn_stop = RoundedButton(
            content,
            text="STOP CAMERA",
            command=self.stop_camera,
            bg_color="#D32F2F",
            width=258,
            height=50,
        )
        # Nút stop ẩn mặc định

    def create_status_badge(self):
        # Tạo một frame nhỏ "nổi" ở góc trên bên phải của vùng video
        self.status_frame = tk.Frame(
            self.right_panel, bg=THEME["surface"], padx=10, pady=5
        )
        # Dùng place để đặt vị trí tuyệt đối (tương đối theo cha)
        self.status_frame.place(relx=0.98, y=20, anchor="ne")

        self.status_dot = tk.Canvas(
            self.status_frame,
            width=10,
            height=10,
            bg=THEME["surface"],
            highlightthickness=0,
        )
        self.status_dot.pack(side=tk.LEFT, padx=(0, 5))
        self.dot_id = self.status_dot.create_oval(
            0, 0, 10, 10, fill="#9E9E9E", outline=""
        )

        self.status_text = tk.Label(
            self.status_frame, text="Ready", bg=THEME["surface"], font=("Segoe UI", 9)
        )
        self.status_text.pack(side=tk.LEFT)

    def add_toggle(self, parent, text, var):
        row = tk.Frame(parent, bg=THEME["surface"])
        row.pack(fill=tk.X, pady=5)
        tk.Label(row, text=text, bg=THEME["surface"], font=("Segoe UI", 10)).pack(
            side=tk.LEFT
        )
        MaterialSwitch(row, var, self.update_settings).pack(side=tk.RIGHT)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if path:
            success, msg, img = self.app.load_source_image(path)
            if success:
                # Hiển thị preview
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                # Resize cover/contain logic
                w, h = 260, 180
                pil.thumbnail((w, h), Image.Resampling.LANCZOS)
                tk_img = ImageTk.PhotoImage(pil)
                self.preview_label.config(image=tk_img, text="")
                self.preview_label.image = tk_img
                self.update_status("#4CAF50", "Source Loaded")
            else:
                messagebox.showerror("Error", msg)

    def update_settings(self):
        self.app.face_detection_enabled = self.var_detect.get()
        self.app.face_swap_enabled = self.var_swap.get()
        self.app.mouth_blend_enabled = self.var_blend.get()

    def update_status(self, color, text):
        self.status_dot.itemconfig(self.dot_id, fill=color)
        self.status_text.config(text=text)

    def start_camera(self):
        if self.app.start_camera()[0]:
            self.btn_start.pack_forget()
            self.btn_stop.pack(side=tk.BOTTOM)
            self.update_status("#2196F3", "Live Processing")
            self.update_video()

    def stop_camera(self):
        self.app.stop_camera()
        self.btn_stop.pack_forget()
        self.btn_start.pack(side=tk.BOTTOM)
        self.video_label.config(image="", text="Camera Stopped")
        self.update_status("#9E9E9E", "Stopped")

    def update_video(self):
        if not self.app.is_running:
            return
        frame = self.app.get_current_frame()
        if frame is not None:
            # Lấy kích thước thực tế của vùng video bên phải
            w = self.video_label.winfo_width()
            h = self.video_label.winfo_height()

            if w > 1 and h > 1:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # Logic "Contain" (Giữ tỉ lệ, có viền đen nếu khác tỉ lệ)
                # để xem toàn bộ khung hình camera
                img_ratio = img.width / img.height
                target_ratio = w / h

                if img_ratio > target_ratio:
                    new_w = w
                    new_h = int(w / img_ratio)
                else:
                    new_h = h
                    new_w = int(h * img_ratio)

                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                # Để làm cho nó "tràn viền" kiểu Cover (cắt hình), bạn có thể đổi logic trên.
                # Nhưng với camera webcam, thường ta muốn thấy hết hình (Contain).
                # Vì background là màu đen (video_bg), nó sẽ hòa vào nhau.

                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.config(image=imgtk, text="")
                self.video_label.image = imgtk

        self.root.after(30, self.update_video)

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.app.cleanup)
        self.root.mainloop()


def run(app):
    FaceSwapUI(app).run()
