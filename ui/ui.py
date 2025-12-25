import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
from PIL import Image, ImageTk

# Import App
from modules.app import FaceSwapApp

# --- THEME ---
THEME = {
    "primary": "#2196F3",
    "primary_dark": "#1976D2",
    "bg_left": "#F5F7FA",
    "bg_right": "#101010",
    "surface": "#FFFFFF",
    "text_main": "#263238",
    "text_sub": "#78909C",
    "border": "#E0E0E0",
    "danger": "#FF5252",
    "success": "#4CAF50",
    "fps_bg": "rgba(0,0,0,0.5)",  # Giả lập màu nền bán trong suốt
}


class RoundedButton(tk.Canvas):
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
            bg=THEME["bg_left"],
            highlightthickness=0,
        )
        self.command = command
        self.create_rounded_rect(0, 0, width, height, height / 2, fill=bg_color)
        self.create_text(
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
    def __init__(self, parent, variable, command=None, bg_color=THEME["surface"]):
        super().__init__(
            parent,
            width=46,
            height=24,
            bg=bg_color,
            highlightthickness=0,
            cursor="hand2",
        )
        self.variable = variable
        self.command = command
        self.bind("<Button-1>", self.toggle)
        self.render()

    def render(self):
        self.delete("all")
        is_on = self.variable.get()
        color = THEME["primary"] if is_on else "#CFD8DC"
        self.create_polygon(
            [10, 4, 36, 4, 36, 20, 10, 20],
            smooth=True,
            width=16,
            outline=color,
            fill=color,
        )
        x = 34 if is_on else 12
        self.create_oval(x - 9, 3, x + 9, 21, fill="white", outline="")

    def toggle(self, event=None):
        self.variable.set(not self.variable.get())
        self.render()
        if self.command:
            self.command()


class MainUI:
    def __init__(self):
        self.app = FaceSwapApp()
        self.root = tk.Tk()
        self.setup_window()
        self.create_layout()

    def run(self):
        self.update_ui_loop()
        self.root.mainloop()

    def setup_window(self):
        self.root.title("Face Swap AI Pro")
        w, h = 1100, 720
        x = (self.root.winfo_screenwidth() - w) // 2
        y = (self.root.winfo_screenheight() - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        self.root.configure(bg=THEME["bg_left"])
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_layout(self):
        container = tk.Frame(self.root, bg=THEME["bg_left"])
        container.pack(fill=tk.BOTH, expand=True)

        # === LEFT PANEL ===
        left_panel = tk.Frame(container, bg=THEME["bg_left"], width=340)
        left_panel.pack(side=tk.LEFT, fill=tk.Y)
        left_panel.pack_propagate(False)

        content = tk.Frame(left_panel, bg=THEME["bg_left"])
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Header
        tk.Label(
            content,
            text="Face Swap AI",
            font=("Segoe UI", 18, "bold"),
            bg=THEME["bg_left"],
            fg=THEME["primary"],
        ).pack(anchor="w", pady=(0, 20))

        # Preview
        tk.Label(
            content,
            text="Source Face",
            font=("Segoe UI", 10, "bold"),
            bg=THEME["bg_left"],
            fg=THEME["text_sub"],
        ).pack(anchor="w", pady=(0, 5))

        self.preview_box = tk.Label(
            content, bg="#ECEFF1", text="No Image", fg=THEME["text_sub"]
        )
        self.preview_box.pack(fill=tk.X, ipady=40, pady=(0, 15))

        RoundedButton(
            content,
            text="UPLOAD PHOTO",
            width=260,
            height=40,
            bg_color="#E3F2FD",
            text_color=THEME["primary"],
            command=self.handle_upload,
        ).pack()

        tk.Frame(content, bg="#CFD8DC", height=1).pack(fill=tk.X, pady=20)

        # Settings
        tk.Label(
            content,
            text="Settings",
            font=("Segoe UI", 10, "bold"),
            bg=THEME["bg_left"],
            fg=THEME["text_sub"],
        ).pack(anchor="w", pady=(0, 10))

        self.settings_frame = tk.Frame(content, bg=THEME["bg_left"])
        self.settings_frame.pack(fill=tk.X)
        self._build_settings_grid()

        # Buttons
        tk.Frame(content, bg=THEME["bg_left"]).pack(fill=tk.BOTH, expand=True)
        self.btn_start = RoundedButton(
            content,
            text="START CAMERA",
            width=280,
            height=50,
            command=self.handle_start,
            bg_color=THEME["primary"],
        )
        self.btn_start.pack(side=tk.BOTTOM, pady=10)

        self.btn_stop = RoundedButton(
            content,
            text="STOP CAMERA",
            width=280,
            height=50,
            command=self.handle_stop,
            bg_color=THEME["danger"],
        )

        # === RIGHT PANEL ===
        right_panel = tk.Frame(container, bg=THEME["bg_right"])
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(
            right_panel,
            bg=THEME["bg_right"],
            text="Camera Off",
            fg="#555",
            font=("Segoe UI", 24),
        )
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # STATUS BADGE (Góc Trái)
        self.status_badge = tk.Label(
            right_panel,
            text="Ready",
            bg="black",
            fg="white",
            font=("Segoe UI", 9),
            padx=10,
            pady=5,
        )
        self.status_badge.place(relx=0.02, rely=0.02, anchor="nw")

        # FPS BADGE (Góc Phải) - Mới thêm vào
        self.fps_badge = tk.Label(
            right_panel,
            text="FPS: 0.0",
            bg="black",
            fg="#00E676",
            font=("Courier New", 10, "bold"),
            padx=10,
            pady=5,
        )
        self.fps_badge.place(relx=0.98, rely=0.02, anchor="ne")

    def _build_settings_grid(self):
        self.vars = {}
        features = [
            ("Show Box", "bounding_box", False),
            ("Swap", "swap", True),
            ("Mouth Mask", "mouth_mask", True),
            ("Enhance", "enhance", False),
        ]
        for i, (label, key, val) in enumerate(features):
            row, col = i // 2, i % 2
            frame = tk.Frame(self.settings_frame, bg=THEME["surface"], padx=8, pady=6)
            frame.grid(row=row, column=col, padx=5, pady=5, sticky="ew")
            self.settings_frame.grid_columnconfigure(col, weight=1)

            tk.Label(
                frame,
                text=label,
                bg=THEME["surface"],
                font=("Segoe UI", 9, "bold"),
                fg=THEME["text_main"],
            ).pack(anchor="w")

            var = tk.BooleanVar(value=val)
            self.vars[key] = var
            MaterialSwitch(
                frame,
                var,
                command=lambda k=key: self.app.update_setting(k, self.vars[k].get()),
            ).pack(anchor="e", pady=(2, 0))

    def handle_upload(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if path:
            success, msg, img = self.app.set_source_image(path)
            if success:
                self._show_preview(img)
                self.status_badge.config(text="Source Loaded", bg=THEME["success"])
            else:
                messagebox.showerror("Error", msg)

    def handle_start(self):
        success, msg = self.app.start_camera()
        if success:
            self.btn_start.pack_forget()
            self.btn_stop.pack(side=tk.BOTTOM, pady=10)
            self.status_badge.config(text="● Live Processing", fg=THEME["primary"])
        else:
            messagebox.showerror("Error", msg)

    def handle_stop(self):
        self.app.stop_camera()
        self.btn_stop.pack_forget()
        self.btn_start.pack(side=tk.BOTTOM, pady=10)
        self.video_label.config(image="", text="Camera Stopped")
        self.status_badge.config(text="Stopped", fg="white")
        self.fps_badge.config(text="FPS: 0.0")  # Reset FPS display

    def on_close(self):
        self.app.stop_camera()
        self.root.destroy()

    def _show_preview(self, cv2_img):
        img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        im_pil.thumbnail((260, 180))
        imgtk = ImageTk.PhotoImage(im_pil)
        self.preview_box.config(image=imgtk, text="", height=0)
        self.preview_box.image = imgtk

    def update_ui_loop(self):
        frame = self.app.get_frame()

        # Cập nhật FPS từ App Logic
        current_fps = self.app.get_fps()
        if self.app.is_running:
            self.fps_badge.config(text=f"FPS: {current_fps}")

        if frame is not None:
            w = self.video_label.winfo_width()
            h = self.video_label.winfo_height()
            if w > 1 and h > 1:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(img)

                img_ratio = im_pil.width / im_pil.height
                win_ratio = w / h
                if img_ratio > win_ratio:
                    new_w = w
                    new_h = int(w / img_ratio)
                else:
                    new_h = h
                    new_w = int(h * img_ratio)

                im_pil = im_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(im_pil)
                self.video_label.config(image=imgtk, text="")
                self.video_label.image = imgtk

        self.root.after(30, self.update_ui_loop)
