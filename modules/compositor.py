import numpy as np
import cv2
from gfpgan import GFPGANer


class Compositor:
    def __init__(self, device):
        self.gfpgan = GFPGANer(
            model_path="models/GFPGANv1.4.pth",
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device = device
        )

    def enhance(self, frame):
        _, _, restored = self.gfpgan.enhance(
            frame,
            has_aligned=False,
            only_center_face=False,
            paste_back=True
        )
        return restored

    def blend_mouth_mask(self, frame, swapped_frame, face):
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

        return final_frame
