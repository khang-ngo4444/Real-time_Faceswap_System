import numpy as np
import cv2
from gfpgan import GFPGANer
import threading
import queue
import time

class Compositor:
    def __init__(self, gfpgan_model_path="models/GFPGANv1.4.pth", verbose=False):
        self.prev_frame = None
        self.verbose = verbose
        self.gfpgan = None
        
        # === OPTIMIZATION: Enhancement Queue ===
        self.enhance_queue = queue.Queue(maxsize=1)  # Chỉ giữ 1 frame mới nhất
        self.enhanced_cache = None
        self.enhancement_lock = threading.Lock()
        self.enhance_thread_running = False
        self.last_enhance_time = 0
        self.enhance_interval = 0.7  # Enhance mỗi 400ms (2.5 FPS)
        
        if GFPGANer is not None:
            try:
                self.gfpgan = GFPGANer(
                    model_path=gfpgan_model_path,
                    upscale=1,
                    arch="clean",
                    channel_multiplier=2,
                    bg_upsampler=None,
                )
                if self.verbose:
                    print("[GFPGAN] Loaded")
                
                # Khởi động enhancement thread
                self.enhance_thread_running = True
                threading.Thread(target=self._enhancement_worker, daemon=True).start()
            except Exception as e:
                if self.verbose:
                    print(f"[GFPGAN] Load failed: {e}")

    def _enhancement_worker(self):
        """Worker thread xử lý GFPGAN không đồng bộ"""
        while self.enhance_thread_running:
            try:
                # Lấy frame từ queue (timeout để check thread status)
                frame = self.enhance_queue.get(timeout=0.1)
                
                # Chạy GFPGAN
                _, _, restored = self.gfpgan.enhance(
                    frame,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True,
                )
                
                # Cập nhật cache
                with self.enhancement_lock:
                    self.enhanced_cache = restored
                    
            except queue.Empty:
                continue
            except Exception as e:
                if self.verbose:
                    print(f"[GFPGAN] Worker error: {e}")

    def enhance(self, frame):
        """
        OPTIMIZED: Enhance async với cache fallback
        - Nếu có cache → return cache (fast)
        - Submit frame mới vào queue để enhance background
        - Không block main thread
        """
        if self.gfpgan is None:
            return frame
        
        current_time = time.time()
        
        # Throttle: Chỉ submit frame mới sau enhance_interval
        if current_time - self.last_enhance_time >= self.enhance_interval:
            try:
                # Clear queue cũ và submit frame mới (chỉ giữ frame mới nhất)
                while not self.enhance_queue.empty():
                    try:
                        self.enhance_queue.get_nowait()
                    except queue.Empty:
                        break
                
                self.enhance_queue.put_nowait(frame.copy())
                self.last_enhance_time = current_time
            except queue.Full:
                pass  # Queue full, skip frame
        
        # Return cached enhanced frame hoặc original nếu chưa có cache
        with self.enhancement_lock:
            if self.enhanced_cache is not None:
                return self.enhanced_cache
        
        return frame  # Fallback: chưa enhance lần nào

    def create_box_mask(self, frame, face):
        """MASK LOẠI 1: Box Mask - Vùng chữ nhật quanh mặt"""
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        bbox = cv2.boundingRect(face.landmark_2d_106.astype(np.int32))
        x, y, bw, bh = bbox
        
        padding = [10, 10, 10, 10]
        x1 = max(0, x - padding[3])
        y1 = max(0, y - padding[0])
        x2 = min(w, x + bw + padding[1])
        y2 = min(h, y + bh + padding[2])
        
        mask[y1:y2, x1:x2] = 255
        return mask
    
    def create_occlusion_mask(self, frame, face, parsing_mask):
        """MASK LOẠI 2: Occlusion Mask - với Motion + Depth + Color"""
        h, w = frame.shape[:2]
        occl_mask = np.zeros((h, w), dtype=np.uint8)
        
        bbox = cv2.boundingRect(face.landmark_2d_106.astype(np.int32))
        x, y, bw, bh = bbox
        pad = int(max(bw, bh) * 0.5)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)
        
        if x2 <= x1 or y2 <= y1:
            return occl_mask
        
        face_roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Motion detection
        motion_mask = np.zeros(gray.shape, dtype=np.uint8)
        if self.prev_frame is not None:
            try:
                prev_roi = self.prev_frame[y1:y2, x1:x2]
                if prev_roi.shape == face_roi.shape:
                    prev_gray = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)
                    frame_diff = cv2.absdiff(gray, prev_gray)
                    _, motion_mask = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
                    motion_mask = cv2.dilate(motion_mask, np.ones((7,7), np.uint8), iterations=2)
            except:
                pass
        
        # Edge detection
        edges = cv2.Canny(gray, 20, 80)
        kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, kernel_edge, iterations=1)
        
        # Depth estimation
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = np.abs(laplacian)
        lap_var = cv2.normalize(lap_var, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, depth_mask = cv2.threshold(lap_var, 30, 255, cv2.THRESH_BINARY)
        
        # Skin color
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        non_skin = cv2.bitwise_not(skin_mask)
        
        # Contours
        contour_mask = np.zeros(gray.shape, dtype=np.uint8)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 100 < area < (face_roi.shape[0] * face_roi.shape[1] * 0.3):
                cv2.drawContours(contour_mask, [cnt], -1, 255, -1)
        
        # Combine signals
        occl_roi = cv2.addWeighted(edges, 0.3, motion_mask, 0.2, 0)
        occl_roi = cv2.addWeighted(occl_roi, 1.0, depth_mask, 0.2, 0)
        occl_roi = cv2.addWeighted(occl_roi, 1.0, non_skin, 0.2, 0)
        occl_roi = cv2.addWeighted(occl_roi, 1.0, contour_mask, 0.1, 0)
        
        _, occl_roi = cv2.threshold(occl_roi, 80, 255, cv2.THRESH_BINARY)
        
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        occl_roi = cv2.morphologyEx(occl_roi, cv2.MORPH_CLOSE, kernel_close, iterations=3)
        
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        occl_roi = cv2.dilate(occl_roi, kernel_dilate, iterations=2)
        
        if parsing_mask is not None:
            parse_roi = parsing_mask[y1:y2, x1:x2]
            potential_face = (parse_roi > 0).astype(np.uint8) * 255
            kernel_expand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            potential_face = cv2.dilate(potential_face, kernel_expand, iterations=2)
            occl_roi = cv2.bitwise_and(occl_roi, potential_face)
        
        occl_mask[y1:y2, x1:x2] = occl_roi
        self.prev_frame = frame.copy()
        
        return occl_mask
    
    def create_region_mask(self, parsing_mask, exclude_regions=['hair', 'ears']):
        """MASK LOẠI 3: Region Mask"""
        if parsing_mask is None:
            return None
        
        h, w = parsing_mask.shape[:2]
        region_mask = np.zeros((h, w), dtype=np.uint8)
        
        keep_classes = [1, 2, 3, 4, 5, 10, 11, 12, 13]
        
        for cls in keep_classes:
            region_mask[parsing_mask == cls] = 255
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        region_mask = cv2.erode(region_mask, kernel, iterations=1)
        
        return region_mask
    
    def create_mouth_mask(self, frame, face):
        """Mouth mask riêng biệt"""
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        landmarks = face.landmark_2d_106.astype(np.int32)
        lower_lip_order = [64,63,67,68,69,18,19,20,21,22,23,24,0,8,7,6,5,4,3,2,65]
        lower_lip_landmarks = landmarks[lower_lip_order].astype(np.int32)
        hull = cv2.convexHull(lower_lip_landmarks)
        cv2.fillConvexPoly(mask, hull, 255)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        mask = cv2.dilate(mask, kernel, 1)
        
        return mask
    
    def combine_masks(self, box_mask, occlusion_mask, region_mask, mouth_mask, blur_amount=0.3):
        """Kết hợp các mask"""
        h, w = box_mask.shape[:2]
        
        if region_mask is not None:
            final_mask = region_mask.copy()
        else:
            final_mask = box_mask.copy()
        
        if occlusion_mask is not None:
            kernel_safe = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            occl_expanded = cv2.dilate(occlusion_mask, kernel_safe, iterations=2)
            final_mask = cv2.bitwise_and(final_mask, cv2.bitwise_not(occl_expanded))
        
        final_mask = cv2.bitwise_and(final_mask, cv2.bitwise_not(mouth_mask))
        final_mask = cv2.bitwise_and(final_mask, box_mask)
        
        blur_kernel_size = int(35 * (blur_amount / 0.3))
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1
        final_mask = cv2.GaussianBlur(final_mask, (blur_kernel_size, blur_kernel_size), 0)
        
        return final_mask
    
    def blend_composite(self, frame, swapped_frame, face, parsing_mask, blur_amount=0.5):
        """Blend chính"""
        h, w = frame.shape[:2]
        
        box_mask = self.create_box_mask(frame, face)
        occlusion_mask = self.create_occlusion_mask(frame, face, parsing_mask)
        region_mask = self.create_region_mask(parsing_mask)
        mouth_mask = self.create_mouth_mask(frame, face)
        
        final_mask = self.combine_masks(
            box_mask, 
            occlusion_mask, 
            region_mask, 
            mouth_mask,
            blur_amount=blur_amount
        )

        # Blend 2 bước
        mouth_mask_blur = cv2.GaussianBlur(mouth_mask, (31, 31), 0)
        mouth_mask_3ch = cv2.cvtColor(mouth_mask_blur, cv2.COLOR_GRAY2BGR) / 255.0
        mouth_blended = (frame * mouth_mask_3ch + swapped_frame * (1 - mouth_mask_3ch)).astype(np.uint8)
        
        mask_3ch = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR) / 255.0
        output = (swapped_frame * mask_3ch + mouth_blended * (1 - mask_3ch)).astype(np.uint8)

        return output
    
    def __del__(self):
        """Cleanup khi object bị destroy"""
        self.enhance_thread_running = False