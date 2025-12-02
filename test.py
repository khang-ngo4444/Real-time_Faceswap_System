import cv2
from insightface.app import FaceAnalysis

# Initialize the model
model = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider'])
model.prepare(ctx_id=0, det_size=(640, 640))
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    faces = model.get(frame)

    for face in faces:
        if face.bbox is None:
            print("Warning: bbox=None")
            continue

        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        if face.landmark_2d_106 is None:
            print("Warning: landmark=None")
            continue

        for x, y in face.landmark_2d_106.astype(int):
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    cv2.imshow("Preview", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
