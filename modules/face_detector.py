from insightface.app import FaceAnalysis

class FaceDetector:
    def __init__(self):
        self.model = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
        self.model.prepare(ctx_id=0)

    def detect(self, img):
        return self.model.get(img)