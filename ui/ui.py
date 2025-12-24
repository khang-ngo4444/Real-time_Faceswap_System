import tkinter as tk
from tkinter import filedialog, messagebox

import cv2  # Chỉ dùng để convert màu hiển thị, không dùng logic AI

# Import Logic Class
from app import FaceSwapApp
from PIL import Image, ImageTk

# --- THEME CONFIG ---
THEME = {
    "primary": "#2196F3",
    "primary_dark": "#1976D2",
    "bg_left": "#F5F7FA",  # Xám rất nhạt sang trọng
    "bg_right": "#000000",  # Đen tuyền cho video
    "surface": "#FFFFFF",
    "text_main": "#1A1A1A",
    "text_sub": "#6C757D",
    "border": "#E9ECEF",
    "danger": "#EF5350",
}


class RoundedButton(tk.Canvas):
    """Custom Widget: Nút bo tròn kiểu Flutter"""

    def __init__(
        self,
        parent,
        text,
        command=None,
        width=200,
        height=45,
        bg_color=THEME["primary"],
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
        self.bg_normal = bg_color
        self.rect = self.create_rounded_rect(
            0, 0, width, height, height / 2, fill=bg_color
        )
        self.text = self.create_text(
            width / 2,
            height / 2,
            text=text,
            fill=text_color,
            font=("Segoe UI", 10, "bold"),
        )
        self.bind("<Button-1>", lambda e: self.command() if self.command else None)
        self.configure(cursor="hand2")

    def create_rounded_rect(self, x1, y1, x2, y2, r, **kwargs):
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
    """Custom Widget: Switch gạt tắt/bật"""

    def __init__(self, parent, variable, command=None):
        super().__init__(
            parent,
            width=46,
            height=24,
            bg=THEME["surface"],
            highlightthickness=0,
            cursor="hand2",
        )
        self.variable = variable
        self.command = command
        self.is_on = variable.get()
        self.bind("<Button-1>", self.toggle)
        self.render()

    def render(self):
        self.delete("all")
        color = THEME["primary"] if self.is_on else "#B0BEC5"
        self.create_polygon(
            [10, 4, 36, 4, 36, 20, 10, 20],
            smooth=True,
            width=16,
            outline=color,
            fill=color,
        )
        x = 34 if self.is_on else 12
        self.create_oval(x - 10, 2, x + 10, 22, fill="white", outline="")

    def toggle(self, event=None):
        self.is_on = not self.is_on
        self.variable.set(self.is_on)
        self.render()
        if self.command:
            self.command()


class MainUI:
    def __init__(self):
        # 1. Khởi tạo Logic Core
        self.app = FaceSwapApp()

        # 2. Setup Window
        self.root = tk.Tk()
        self.setup_window()
        self.create_layout()

        # 3. Bắt đầu vòng lặp update UI
        self.update_ui_loop()

    def setup_window(self):
        self.root.title("Face Swap AI")
        w, h = 1200, 720
        x = (self.root.winfo_screenwidth() - w) // 2
        y = (self.root.winfo_screenheight() - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        self.root.configure(bg=THEME["bg_left"])

    def create_layout(self):
        # Layout chính: Chia 2 cột (Left Control | Right Video)
        container = tk.Frame(self.root, bg=THEME["bg_left"])
        container.pack(fill=tk.BOTH, expand=True)

        # === LEFT PANEL (CONTROLS) ===
        # Width cố định 350px
        left_panel = tk.Frame(container, bg=THEME["bg_left"], width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.Y)
        left_panel.pack_propagate(False)

        # Card container (cái hộp trắng)
        card = tk.Frame(
            left_panel,
            bg=THEME["surface"],
            highlightthickness=1,
            highlightbackground=THEME["border"],
        )
        card.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Nội dung trong Card
        content = tk.Frame(card, bg=THEME["surface"])
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=25)

        # Header Text
        tk.Label(
            content,
            text="Face Swap AI",
            font=("Segoe UI", 18, "bold"),
            bg=THEME["surface"],
            fg=THEME["primary"],
        ).pack(anchor="w", pady=(0, 20))

        # --- PREVIEW ẢNH SOURCE ---
        tk.Label(
            content,
            text="Source Face",
            font=("Segoe UI", 10, "bold"),
            bg=THEME["surface"],
            fg=THEME["text_sub"],
        ).pack(anchor="w", pady=(0, 5))

        self.preview_box = tk.Label(
            content, bg=THEME["bg_left"], text="No Image", fg=THEME["text_sub"]
        )
        self.preview_box.pack(
            fill=tk.X, ipady=40, pady=(0, 15)
        )  # ipady tạo chiều cao giả

        RoundedButton(
            content,
            text="UPLOAD PHOTO",
            width=260,
            height=40,
            bg_color="#E3F2FD",
            text_color=THEME["primary"],
            command=self.handle_upload,
        ).pack()

        tk.Frame(content, bg=THEME["border"], height=1).pack(fill=tk.X, pady=25)

        # --- SETTINGS ---
        tk.Label(
            content,
            text="Settings",
            font=("Segoe UI", 10, "bold"),
            bg=THEME["surface"],
            fg=THEME["text_sub"],
        ).pack(anchor="w", pady=(0, 10))

        self.vars = {
            "detect": tk.BooleanVar(value=True),
            "swap": tk.BooleanVar(value=True),
            "blend": tk.BooleanVar(value=True),
        }
        self.add_setting_row(content, "Face Detection", "detect")
        self.add_setting_row(content, "Face Swapping", "swap")
        self.add_setting_row(content, "Mouth Blending", "blend")

        # Spacer (Đẩy nút start xuống đáy)
        tk.Frame(content, bg=THEME["surface"]).pack(fill=tk.BOTH, expand=True)

        # --- ACTION BUTTONS ---
        self.btn_start = RoundedButton(
            content,
            text="START CAMERA",
            width=260,
            height=50,
            command=self.handle_start,
        )
        self.btn_start.pack(side=tk.BOTTOM)

        self.btn_stop = RoundedButton(
            content,
            text="STOP CAMERA",
            width=260,
            height=50,
            bg_color=THEME["danger"],
            command=self.handle_stop,
        )
        # Nút stop ẩn đi ban đầu

        # === RIGHT PANEL (FULL VIDEO) ===
        right_panel = tk.Frame(container, bg=THEME["bg_right"])
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(
            right_panel,
            bg=THEME["bg_right"],
            text="Camera Off",
            fg="#333",
            font=("Segoe UI", 24),
        )
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Status badge nhỏ
        self.lbl_status = tk.Label(
            right_panel,
            text="Ready",
            bg="black",
            fg="white",
            font=("Segoe UI", 9),
            padx=10,
            pady=5,
        )
        self.lbl_status.place(relx=0.02, rely=0.02, anchor="nw")

    def add_setting_row(self, parent, text, key):
        row = tk.Frame(parent, bg=THEME["surface"])
        row.pack(fill=tk.X, pady=6)
        tk.Label(
            row,
            text=text,
            bg=THEME["surface"],
            fg=THEME["text_main"],
            font=("Segoe UI", 10),
        ).pack(side=tk.LEFT)
        MaterialSwitch(
            row,
            self.vars[key],
            command=lambda: self.app.update_setting(key, self.vars[key].get()),
        ).pack(side=tk.RIGHT)

    # --- HANDLERS (GỌI SANG APP.PY) ---

    def handle_upload(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if not path:
            return

        # Gọi Logic: Load ảnh
        success, msg, img_data = self.app.set_source_image(path)

        if success:
            # UI: Hiển thị preview
            self.show_preview(img_data)
            self.lbl_status.config(text="Source Loaded", bg="#4CAF50")
        else:
            messagebox.showerror("Error", msg)

    def handle_start(self):
        # Gọi Logic: Start Cam
        success, msg = self.app.start_camera()
        if success:
            self.btn_start.pack_forget()
            self.btn_stop.pack(side=tk.BOTTOM)
            self.lbl_status.config(text="● Live Processing", fg="#2196F3")
        else:
            messagebox.showerror("Error", msg)

    def handle_stop(self):
        # Gọi Logic: Stop Cam
        self.app.stop_camera()
        self.btn_stop.pack_forget()
        self.btn_start.pack(side=tk.BOTTOM)
        self.video_label.config(image="", text="Camera Stopped")
        self.lbl_status.config(text="Stopped", fg="white")

    # --- VIEW HELPERS ---

    def show_preview(self, cv2_img):
        # Chuyển đổi màu cho UI
        img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)

        # Resize nhỏ để vừa ô preview (Contain style)
        im_pil.thumbnail((260, 180))

        imgtk = ImageTk.PhotoImage(im_pil)
        self.preview_box.config(
            image=imgtk, text="", height=0, width=0
        )  # Reset text/size
        self.preview_box.image = imgtk

    def update_ui_loop(self):
        """Vòng lặp UI độc lập, chỉ lấy frame từ Logic để vẽ"""
        frame = self.app.get_frame()

        if frame is not None:
            # Lấy kích thước hiện tại của vùng hiển thị video
            win_w = self.video_label.winfo_width()
            win_h = self.video_label.winfo_height()

            if win_w > 1 and win_h > 1:
                # Convert màu
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(img)

                # Logic Full Bleed (Contain - Hiển thị toàn bộ hình)
                # Tính toán resize sao cho hình nằm trọn trong khung đen bên phải
                img_ratio = im_pil.width / im_pil.height
                win_ratio = win_w / win_h

                if img_ratio > win_ratio:
                    new_w = win_w
                    new_h = int(win_w / img_ratio)
                else:
                    new_h = win_h
                    new_w = int(win_h * img_ratio)

                im_pil = im_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(im_pil)

                self.video_label.config(image=imgtk, text="")
                self.video_label.image = imgtk

        # Gọi lại sau 30ms (~30 FPS)
        self.root.after(30, self.update_ui_loop)

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()

    def on_close(self):
        self.app.stop_camera()
        self.root.destroy()


if __name__ == "__main__":
    MainUI().run()
