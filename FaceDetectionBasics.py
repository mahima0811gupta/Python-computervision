import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.7)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    # print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection)
            # print(id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic, = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, bbox, (153, 255, 20), 2)
            cv2.putText(img, f'Accurary: {int(detection.score[0]*100)}%',
                        (bbox[0], bbox[1]-15), cv2.FONT_HERSHEY_DUPLEX,
                        0.5, (128, 242, 69), 1)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'Fps: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_DUPLEX,
                1, (128, 242, 69), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)