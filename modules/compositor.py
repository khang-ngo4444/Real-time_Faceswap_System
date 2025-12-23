import numpy as np
import cv2

class Compositor:
    def __init__(self):
        pass

    def detect_occlusion_mask(self, frame, face, parsing_mask):
        """
        PHƯƠNG PHÁP MỚI: Edge-based occlusion detection
        - Dùng edge density thay vì skin detection
        - Ít phụ thuộc ánh sáng hơn
        """
        h, w = frame.shape[:2]
        occl_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Lấy bounding box của face
        bbox = cv2.boundingRect(face.landmark_2d_106.astype(np.int32))
        x, y, bw, bh = bbox
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + bw), min(h, y + bh)
        
        if x2 <= x1 or y2 <= y1:
            return occl_mask
        
        face_roi = frame[y1:y2, x1:x2]
        
        # ========== PHƯƠNG PHÁP 1: EDGE DENSITY ==========
        # Vùng có tay/vật thể → nhiều edge hơn da mặt phẳng
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Tính mật độ edge trong các block 20x20
        block_size = 20
        rh, rw = face_roi.shape[:2]
        edge_density = np.zeros((rh, rw), dtype=np.float32)
        
        for i in range(0, rh, block_size):
            for j in range(0, rw, block_size):
                block = edges[i:i+block_size, j:j+block_size]
                if block.size > 0:
                    density = np.count_nonzero(block) / float(block.size)
                    edge_density[i:i+block_size, j:j+block_size] = density
        
        # Threshold: vùng nào edge_density > 0.15 → có vật che
        occl_roi = (edge_density > 0.15).astype(np.uint8) * 255
        
        # ========== KẾT HỢP PARSING MASK ==========
        if parsing_mask is not None:
            parse_roi = parsing_mask[y1:y2, x1:x2]
            face_parse_mask = (parse_roi > 0).astype(np.uint8) * 255
        else:
            hull = cv2.convexHull(face.landmark_2d_106.astype(np.int32))
            face_parse_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(face_parse_mask, hull, 255)
            face_parse_mask = face_parse_mask[y1:y2, x1:x2]
        
        # Chỉ giữ occlusion trong vùng face
        occl_roi = cv2.bitwise_and(occl_roi, face_parse_mask)
        
        # Làm sạch noise nhẹ
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        occl_roi = cv2.morphologyEx(occl_roi, cv2.MORPH_CLOSE, kernel)
        occl_roi = cv2.morphologyEx(occl_roi, cv2.MORPH_OPEN, kernel)
        
        # ========== SAFETY CHECK ==========
        face_area = np.count_nonzero(face_parse_mask)
        occl_area = np.count_nonzero(occl_roi)
        
        if face_area > 0:
            occl_ratio = occl_area / float(face_area)
            print(f"[OCCL] Edge-based occlusion ratio: {occl_ratio:.2%}")
            
            # Nếu >50% bị detect → có thể sai, bỏ qua
            if occl_ratio > 0.5:
                print("[OCCL] Ratio quá cao, skip occlusion")
                return occl_mask
        
        occl_mask[y1:y2, x1:x2] = occl_roi
        return occl_mask

    def blend_mouth_mask(self, frame, swapped_frame, face):
        """
        LUỒNG RIÊNG: Chỉ xử lý mouth mask (code bạn muốn giữ)
        """
        h, w, _ = frame.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        landmarks = face.landmark_2d_106.astype(np.int32)
        lower_lip_order = [64,63,67,68,69,18,19,20,21,22,23,24,0,8,7,6,5,4,3,2,65]
        lower_lip_landmarks = landmarks[lower_lip_order].astype(np.int32)
        hull = cv2.convexHull(lower_lip_landmarks)
        cv2.fillConvexPoly(mask, hull, 255)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        mask = cv2.dilate(mask, kernel, 1)
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0

        final_frame = (
            frame * mask_3ch + swapped_frame * (1 - mask_3ch)
        ).astype(np.uint8)
        
        return final_frame, mask

    def blend_composite(self, frame, swapped_frame, face, parsing_mask):
        """
        LUỒNG TỔNG HỢP: Áp dụng occlusion mask sau khi đã blend mouth
        """
        h, w = frame.shape[:2]
        
        # ========== BƯỚC 1: XỬ LÝ MOUTH (LUỒNG RIÊNG) ==========
        mouth_blended, mouth_mask = self.blend_mouth_mask(frame, swapped_frame, face)
        
        # ========== BƯỚC 2: TẠO FACE AREA MASK ==========
        if parsing_mask is None:
            landmarks = face.landmark_2d_106.astype(np.int32)
            hull = cv2.convexHull(landmarks)
            face_area_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(face_area_mask, hull, 255)
        else:
            face_area_mask = (parsing_mask > 0).astype(np.uint8) * 255

        # Erode nhẹ để tránh artifact ở biên
        kernel_face = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        face_area_mask = cv2.erode(face_area_mask, kernel_face, iterations=1)

        # ========== BƯỚC 3: PHÁT HIỆN OCCLUSION ==========
        occl_mask = self.detect_occlusion_mask(frame, face, parsing_mask)
        
        # Dilate occlusion nhẹ để che kín vật thể
        kernel_occl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        occl_mask = cv2.dilate(occl_mask, kernel_occl, iterations=1)
        
        # ========== BƯỚC 4: TỔNG HỢP MASK CUỐI CÙNG ==========
        # Swap mask = Face Area - Mouth - Occlusion
        swap_mask = cv2.bitwise_and(face_area_mask, cv2.bitwise_not(mouth_mask))
        swap_mask = cv2.bitwise_and(swap_mask, cv2.bitwise_not(occl_mask))
        
        # Blur để blend mượt
        swap_mask = cv2.GaussianBlur(swap_mask, (35, 35), 0)

        # ========== BƯỚC 5: BLEND FINAL ==========
        mask_3ch = cv2.cvtColor(swap_mask, cv2.COLOR_GRAY2BGR) / 255.0
        
        # Blend: mouth_blended (đã có mouth thật) + swapped_frame (phần face swap)
        output = (swapped_frame * mask_3ch + mouth_blended * (1 - mask_3ch)).astype(np.uint8)

        # ========== DEBUG WINDOWS ==========
        cv2.imshow("1_Face Area", face_area_mask)
        cv2.imshow("2_Mouth", mouth_mask)
        cv2.imshow("3_Occlusion", occl_mask)
        cv2.imshow("4_Final Swap", swap_mask)

        return output