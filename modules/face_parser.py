import cv2
import numpy as np
from insightface.model_zoo import get_model

class FaceParser:
    def __init__(self):
        self.model = get_model('bisenet_resnet18_celebamaskhq', providers=['CUDAExecutionProvider'])

    def parse(self, img, face):
        try:
            bbox = face.bbox.astype(int)
            w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
            pad = int(max(w, h) * 0.2)
            x1, y1 = max(0, bbox[0]-pad), max(0, bbox[1]-pad)
            x2, y2 = min(img.shape[1], bbox[2]+pad), min(img.shape[0], bbox[3]+pad)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0: return None
            
            res = self.model.get(crop)[0] # Bisenet output
            if res.dtype == np.float32:
                res = (res * 255).astype(np.uint8)
                
            res = cv2.resize(res, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
            full_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = res
            return full_mask
        except:
            return None