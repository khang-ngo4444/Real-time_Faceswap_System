import numpy as np
import cv2

class Compositor:
    def __init__(self):
        self.xseg_model = None  # Load từ models/xseg_2.onnx nếu có
        # Thêm buffer cho motion detection
        self.prev_frame = None
        
    def create_box_mask(self, frame, face):
        """
        MASK LOẠI 1: Box Mask - Vùng chữ nhật quanh mặt
        """
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        bbox = cv2.boundingRect(face.landmark_2d_106.astype(np.int32))
        x, y, bw, bh = bbox
        
        # Padding mở rộng (top, right, bottom, left)
        padding = [10, 10, 10, 10]
        x1 = max(0, x - padding[3])
        y1 = max(0, y - padding[0])
        x2 = min(w, x + bw + padding[1])
        y2 = min(h, y + bh + padding[2])
        
        mask[y1:y2, x1:x2] = 255
        return mask
    
    def create_occlusion_mask(self, frame, face, parsing_mask):
        """
        MASK LOẠI 2: Occlusion Mask - SUPER IMPROVED với Motion + Depth + Color
        """
        h, w = frame.shape[:2]
        occl_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Lấy bbox mở rộng mạnh để capture tay chạm mặt
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
        
        # ========== 1. MOTION DETECTION (tay đang di chuyển) ==========
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
        
        # ========== 2. EDGE DETECTION (siêu nhạy để bắt ngón tay) ==========
        edges = cv2.Canny(gray, 20, 80) 
        
        # Làm mạnh edges bằng morphology
        kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, kernel_edge, iterations=1)
        
        # ========== 3. DEPTH ESTIMATION (Laplacian Variance - vật gần = blur ít) ==========
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = np.abs(laplacian)
        lap_var = cv2.normalize(lap_var, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Vùng có variance cao = chi tiết cao = gần camera (tay)
        _, depth_mask = cv2.threshold(lap_var, 30, 255, cv2.THRESH_BINARY)
        
        # ========== 4. SKIN COLOR DIFFERENCE (tay thường khác màu mặt khi che) ==========
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        
        # Skin detection
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Tìm vùng NON-SKIN (potential hand)
        non_skin = cv2.bitwise_not(skin_mask)
        
        # ========== 5. CONTOUR DETECTION (phát hiện hình dạng tay) ==========
        contour_mask = np.zeros(gray.shape, dtype=np.uint8)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Chỉ giữ contour vừa phải (tay/ngón tay, không phải noise)
            if 100 < area < (face_roi.shape[0] * face_roi.shape[1] * 0.3):
                cv2.drawContours(contour_mask, [cnt], -1, 255, -1)
        
        # ========== KẾT HỢP TẤT CẢ SIGNALS ==========
        # Weighted combination: edges + motion + depth + non_skin + contours
        occl_roi = cv2.addWeighted(edges, 0.3, motion_mask, 0.2, 0)
        occl_roi = cv2.addWeighted(occl_roi, 1.0, depth_mask, 0.2, 0)
        occl_roi = cv2.addWeighted(occl_roi, 1.0, non_skin, 0.2, 0)
        occl_roi = cv2.addWeighted(occl_roi, 1.0, contour_mask, 0.1, 0)
        
        # Threshold final
        _, occl_roi = cv2.threshold(occl_roi, 80, 255, cv2.THRESH_BINARY)
        
        # Close để fill holes và connect fingers
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        occl_roi = cv2.morphologyEx(occl_roi, cv2.MORPH_CLOSE, kernel_close, iterations=3)
        
        # Dilate để mở rộng vùng occlusion
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        occl_roi = cv2.dilate(occl_roi, kernel_dilate, iterations=2)
        
        # ========== CHỈ GIỮ VÙNG TRONG PARSING MASK (nếu có) ==========
        if parsing_mask is not None:
            parse_roi = parsing_mask[y1:y2, x1:x2]
            # Mở rộng parsing mask để bao cả vùng potential occlusion
            potential_face = (parse_roi > 0).astype(np.uint8) * 255
            kernel_expand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            potential_face = cv2.dilate(potential_face, kernel_expand, iterations=2)
            
            # Chỉ giữ occlusion trong vùng face
            occl_roi = cv2.bitwise_and(occl_roi, potential_face)
        
        occl_mask[y1:y2, x1:x2] = occl_roi
        self.prev_frame = frame.copy()
        
        return occl_mask
    
    def create_region_mask(self, parsing_mask, exclude_regions=['hair', 'ears']):
        """
        MASK LOẠI 3: Region Mask - Chọn vùng cụ thể từ BiSeNet
        
        BiSeNet classes (CelebAMaskHQ):
        0: background
        1: skin
        2: l_brow, 3: r_brow
        4: l_eye, 5: r_eye
        6: eye_g (glasses)
        7: l_ear, 8: r_ear
        9: ear_r (earring)
        10: nose
        11: mouth
        12: u_lip, 13: l_lip
        14: neck
        15: neck_l (necklace)
        16: cloth
        17: hair
        18: hat
        """
        if parsing_mask is None:
            return None
        
        h, w = parsing_mask.shape[:2]
        region_mask = np.zeros((h, w), dtype=np.uint8)
        
        keep_classes = [1, 2, 3, 4, 5, 10, 11, 12, 13]  # Skin, brows, eyes, nose, lips
        
        for cls in keep_classes:
            region_mask[parsing_mask == cls] = 255
        
        # Erode nhẹ để tránh artifact ở biên
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        region_mask = cv2.erode(region_mask, kernel, iterations=1)
        
        return region_mask
    
    def create_mouth_mask(self, frame, face):
        """
        Mouth mask riêng biệt (giữ nguyên miệng thật)
        """
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
    
    def blend_mouth_mask(self, frame, swapped_frame, face):
        """
        Blend miệng thật vào swapped frame
        Dùng create_mouth_mask() + logic blend
        """
        try:
            # Tạo mouth mask từ landmarks
            mask = self.create_mouth_mask(frame, face)
            
            # Blur mask để blend mượt
            mask = cv2.GaussianBlur(mask, (31, 31), 0)
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0

            # Blend: giữ miệng thật từ frame gốc
            final_frame = (
                frame * mask_3ch + swapped_frame * (1 - mask_3ch)
            ).astype(np.uint8)

            return final_frame
        except Exception as e:
            print(f"Error in blend_mouth_mask: {e}")
            return swapped_frame
    
    def combine_masks(self, box_mask, occlusion_mask, region_mask, mouth_mask, blur_amount=0.3):
        """
        Kết hợp các mask theo logic FaceFusion
        
        Logic:
        1. Bắt đầu với Region Mask (BiSeNet) - loại bỏ tóc, tai
        2. Trừ đi Occlusion Mask (vật che) - PRIORITY CAO
        3. Trừ đi Mouth Mask (giữ miệng thật)
        4. Intersection với Box Mask (giới hạn vùng)
        """
        h, w = box_mask.shape[:2]
        
        # Bước 1: Nếu có region_mask → dùng làm base, nếu không → dùng box_mask
        if region_mask is not None:
            final_mask = region_mask.copy()
            print("[MASK] Using Region Mask (BiSeNet) as base")
        else:
            final_mask = box_mask.copy()
            print("[MASK] Using Box Mask as base (no parsing available)")
        
        # Bước 2: Trừ occlusion (vật che) - DILATE THÊM ĐỂ AN TOÀN
        if occlusion_mask is not None:
            kernel_safe = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            occl_expanded = cv2.dilate(occlusion_mask, kernel_safe, iterations=2)
            
            final_mask = cv2.bitwise_and(final_mask, cv2.bitwise_not(occl_expanded))
            occl_pixels = np.count_nonzero(occl_expanded)
            print(f"[MASK] Occlusion removed: {occl_pixels} pixels")
        
        # Bước 3: Trừ mouth mask
        final_mask = cv2.bitwise_and(final_mask, cv2.bitwise_not(mouth_mask))
        mouth_pixels = np.count_nonzero(mouth_mask)
        print(f"[MASK] Mouth excluded: {mouth_pixels} pixels")
        
        # Bước 4: Intersection với box để giới hạn vùng
        final_mask = cv2.bitwise_and(final_mask, box_mask)
        
        # Blur để blend mượt
        blur_kernel_size = int(35 * (blur_amount / 0.3))  # Scale với blur_amount
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1
        final_mask = cv2.GaussianBlur(final_mask, (blur_kernel_size, blur_kernel_size), 0)
        
        return final_mask
    
    def blend_composite(self, frame, swapped_frame, face, parsing_mask, blur_amount=0.5):
        """
        Blend chính theo phong cách FaceFusion
        
        Params:
            blur_amount: 0.3-0.6 (FaceFusion default: 0.3, recommend: 0.5 cho mượt hơn)
        """
        h, w = frame.shape[:2]
        
        # ========== TẠO CÁC MASK ==========
        print("\n========== CREATING MASKS ==========")
        
        box_mask = self.create_box_mask(frame, face)
        print(f"[1] Box Mask: {np.count_nonzero(box_mask)} pixels")
        
        occlusion_mask = self.create_occlusion_mask(frame, face, parsing_mask)
        print(f"[2] Occlusion Mask: {np.count_nonzero(occlusion_mask)} pixels")
        
        region_mask = self.create_region_mask(parsing_mask)
        if region_mask is not None:
            print(f"[3] Region Mask (no hair/ears): {np.count_nonzero(region_mask)} pixels")
        else:
            print("[3] Region Mask: Not available")
        
        mouth_mask = self.create_mouth_mask(frame, face)
        print(f"[4] Mouth Mask: {np.count_nonzero(mouth_mask)} pixels")
        
        # ========== KẾT HỢP MASKS ==========
        final_mask = self.combine_masks(
            box_mask, 
            occlusion_mask, 
            region_mask, 
            mouth_mask,
            blur_amount=blur_amount
        )
        print(f"[FINAL] Combined Swap Mask: {np.count_nonzero(final_mask > 0)} pixels")
        
        # ========== BLEND 2 BƯỚC ==========
        # Bước 1: Blend mouth trước
        mouth_mask_blur = cv2.GaussianBlur(mouth_mask, (31, 31), 0)
        mouth_mask_3ch = cv2.cvtColor(mouth_mask_blur, cv2.COLOR_GRAY2BGR) / 255.0
        mouth_blended = (frame * mouth_mask_3ch + swapped_frame * (1 - mouth_mask_3ch)).astype(np.uint8)
        
        # Bước 2: Blend swap mask lên mouth_blended
        mask_3ch = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR) / 255.0
        output = (swapped_frame * mask_3ch + mouth_blended * (1 - mask_3ch)).astype(np.uint8)
        
        # ========== DEBUG ==========
        cv2.imshow("1_Box", box_mask)
        if region_mask is not None:
            cv2.imshow("2_Region (no hair/ears)", region_mask)
        cv2.imshow("3_Occlusion", occlusion_mask)
        cv2.imshow("4_Mouth", mouth_mask)
        cv2.imshow("5_Final_Swap", final_mask)
        
        return output