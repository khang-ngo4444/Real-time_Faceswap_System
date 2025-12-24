import os.path

from ui.ui import MainUI
from modules.utilities import download

def download_models():
    MODEL_DIR = os.path.join(os.getcwd(), "models")
    MODEL_URLS = [
        "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx",
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
    ]

    download(MODEL_DIR, MODEL_URLS)

def main():
    download_models()

    ui = MainUI()
    ui.run()

if __name__ == "__main__":
    main()
