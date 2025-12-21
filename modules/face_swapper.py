from insightface.model_zoo import get_model


class FaceSwapper:
    def __init__(self, model_path):
        self.model = get_model(model_path, providers=['CUDAExecutionProvider'])

    def swap(self, img, target_face, source_face):
        return self.model.get(img, target_face, source_face, paste_back=True)