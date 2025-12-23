from modules import app
from ui.ui import FaceSwap, run


def main():
    # app.run()

    face_swap = FaceSwap()
    run(face_swap)


if __name__ == "__main__":
    main()
