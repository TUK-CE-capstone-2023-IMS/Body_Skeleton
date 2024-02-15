from cvzone.PoseModule import PoseDetector
import cv2
import socket

# Parameters

width, height = 1280, 720

# Webcam

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Pose Detector
# 객체 detector 초기화

detector = PoseDetector(staticMode=False,
                        modelComplexity=1,
                        smoothLandmarks=True,
                        enableSegmentation=False,
                        smoothSegmentation=True,
                        detectionCon=0.5,
                        trackCon=0.5)

# Communication
# 소켓 생성

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5051)

while True:

    # Get the frame from the webcam
    # 웹캠에서 프레임 가져옴

    success, img = cap.read()

    # Pose
    # 프레임에서 포즈 감지 후 이미지에 다시 표시

    img = detector.findPose(img)

    # Set draw=True to draw the landmarks and bounding box on the image
    # findPosition 메서드를 사용하여 포즈의 랜드마크 위치에 대한 정보 얻음

    lmList, bboxInfo = detector.findPosition(img, draw=True, bboxWithHands=False)

    # Landmark values - (x,y,z)
    # 랜드마크 좌표를 data 리스트에 추가 후 소켓을 통해 서버로 전송

    if lmList:
        data = []
        for lm in lmList:
            data.extend([lm[0], height - lm[1], lm[2]])

        sock.sendto(str.encode(str(data)), serverAddressPort)

    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    cv2.imshow("Image", img)
    cv2.waitKey(1)