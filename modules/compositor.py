import numpy as np
import cv2

class Compositor:
    def __init__(self):
        pass

    def detect_occlusion_mask(self, frame, face, parsing_mask):
        """
        Phát hiện vùng che khuất - CẢI TIẾN: chỉ kích hoạt khi thực sự có vật che
        """
        h, w = frame.shape[:2]
        occl_mask = np.zeros((h, w), dtype=np.uint8)
        
        bbox = cv2.boundingRect(face.landmark_2d_106.astype(np.int32))
        x, y, bw, bh = bbox
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + bw), min(h, y + bh)
        
        if x2 <= x1 or y2 <= y1:
            return occl_mask
        
        face_roi = frame[y1:y2, x1:x2]
        
        # 1. Skin detection - MỞ RỘNG RANGE để tránh false positive
        ycrcb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb)
        # Nới lỏng hơn (trước: 135-180 cho Cr)
        lower_skin = np.array([0, 130, 80], dtype=np.uint8)  # giảm threshold
        upper_skin = np.array([255, 185, 140], dtype=np.uint8)  # tăng threshold
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        
        # 2. Làm sạch skin mask (loại bỏ noise)
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel_clean, iterations=2)
        
        # 3. Face area từ parsing hoặc hull
        if parsing_mask is not None:
            parse_roi = parsing_mask[y1:y2, x1:x2]
            face_parse_mask = (parse_roi > 0).astype(np.uint8) * 255
        else:
            hull = cv2.convexHull(face.landmark_2d_106.astype(np.int32))
            face_parse_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(face_parse_mask, hull, 255)
            face_parse_mask = face_parse_mask[y1:y2, x1:x2]
        
        # 4. Vùng che = non-skin NHƯNG phải có tỷ lệ đủ lớn
        non_skin = cv2.bitwise_not(skin_mask)
        occl_roi = cv2.bitwise_and(non_skin, face_parse_mask)
        
        # **QUAN TRỌNG**: Chỉ giữ lại nếu non-skin < 40% face area (tránh toàn bộ mặt bị đánh là occluded)
        face_area = np.count_nonzero(face_parse_mask)
        non_skin_area = np.count_nonzero(occl_roi)
        
        if face_area > 0:
            non_skin_ratio = non_skin_area / float(face_area)
            print(f"[DEBUG OCCL] Non-skin ratio: {non_skin_ratio:.2%}")
            
            # Nếu >40% là non-skin → có thể do lighting/skin detection sai → BỎ QUA occlusion
            if non_skin_ratio > 0.4:
                print("[DEBUG OCCL] Non-skin ratio quá cao, bỏ qua occlusion detection")
                return occl_mask  # trả về mask rỗng
        
        # Morphological operations để làm sạch noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        occl_roi = cv2.morphologyEx(occl_roi, cv2.MORPH_CLOSE, kernel, iterations=1)  # giảm từ 2→1
        occl_roi = cv2.morphologyEx(occl_roi, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Đưa vào full mask
        occl_mask[y1:y2, x1:x2] = occl_roi
        
        return occl_mask

    def blend_composite(self, frame, swapped_frame, face, parsing_mask):
        """
        frame: Ảnh gốc (chứa miệng thật + tay thật)
        swapped_frame: Ảnh đã swap full mặt
        parsing_mask: Mask từ AI (xác định vùng da mặt, loại bỏ tay)
        """
        h, w = frame.shape[:2]
        
        if parsing_mask is None:
            landmarks = face.landmark_2d_106.astype(np.int32)
            hull = cv2.convexHull(landmarks)
            face_area_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(face_area_mask, hull, 255)
            print("[DEBUG] Parsing mask = None, dùng convex hull")
        else:
            face_area_mask = (parsing_mask > 0).astype(np.uint8) * 255
            print(f"[DEBUG] Parsing mask OK, non-zero pixels: {np.count_nonzero(face_area_mask)}")

        # GIẢM erode để giữ lại nhiều vùng mặt hơn
        kernel_face = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))  # giảm từ 15→9
        face_area_mask = cv2.erode(face_area_mask, kernel_face, iterations=1)

        # --- 2. Xử lý Mouth Mask ---
        mouth_mask = np.zeros((h, w), dtype=np.uint8)
        landmarks = face.landmark_2d_106.astype(np.int32)
        
        lower_lip_order = [64,63,67,68,69,18,19,20,21,22,23,24,0,8,7,6,5,4,3,2,65]
        lip_points = landmarks[lower_lip_order]
        
        cv2.fillConvexPoly(mouth_mask, lip_points, 255)

        # GIẢM dilate của mouth để không ăn quá nhiều vùng mặt
        kernel_mouth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))  # giảm từ 25→20
        mouth_mask = cv2.dilate(mouth_mask, kernel_mouth, iterations=1)  # giảm từ 2→1
        print(f"[DEBUG] Mouth mask non-zero: {np.count_nonzero(mouth_mask)}")

        # --- 3. PHÁT HIỆN CHE KHUẤT ---
        occl_mask = self.detect_occlusion_mask(frame, face, parsing_mask)
        
        # GIẢM dilate của occlusion
        kernel_occl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))  # giảm từ 15→9
        occl_mask = cv2.dilate(occl_mask, kernel_occl, iterations=1)  # giảm từ 2→1
        print(f"[DEBUG] Occlusion mask non-zero: {np.count_nonzero(occl_mask)}")

        # --- 4. TỔNG HỢP MASK ---
        swap_mask = cv2.bitwise_and(face_area_mask, cv2.bitwise_not(mouth_mask))
        swap_mask = cv2.bitwise_and(swap_mask, cv2.bitwise_not(occl_mask))
        print(f"[DEBUG] Final swap_mask non-zero: {np.count_nonzero(swap_mask)}")

        # --- 5. TĂNG OPACITY ---
        swap_mask = cv2.GaussianBlur(swap_mask, (35, 35), 0)

        # --- 6. BLEND ---
        mask_3ch = cv2.cvtColor(swap_mask, cv2.COLOR_GRAY2BGR) / 255.0
        output = (swapped_frame * mask_3ch + frame * (1 - mask_3ch)).astype(np.uint8)

        # DEBUG
        cv2.imshow("1_Face Area", face_area_mask)
        cv2.imshow("2_Mouth", mouth_mask)
        cv2.imshow("3_Occlusion", occl_mask)
        cv2.imshow("4_Final Swap", swap_mask)

        return output