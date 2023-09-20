import cv2
import mediapipe as mp
import time # this is to check the frame rate
from collections import namedtuple

lmData = namedtuple('lmData', ['handID', 'landmarkID', 'x', 'y'])

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionConfidence=0.5, trackConfidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCondifence = detectionConfidence
        self.trackCondifence = trackConfidence

        self.mpHand = mp.solutions.hands
        self.hands = self.mpHand.Hands(self.mode, self.maxHands, 1, self.detectionCondifence, self.trackCondifence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)
        # print(result.multi_hand_landmarks)
        if self.result.multi_hand_landmarks:
            for handLms in self.result.multi_hand_landmarks:  # each element is complete info of one hand
                if draw == True:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHand.HAND_CONNECTIONS)
        return img


    def findPositions(self, img, handID=None, lmID_draw=None, draw=True, drawSize=7): # return landmark info (lm means landmark)
        lmInfo = []
        if handID is None:
            handID = [0, 1]
        if lmID_draw is None:
            lmID = range(21)

        if self.result.multi_hand_landmarks:
            for n, handLms in enumerate(self.result.multi_hand_landmarks):
                for id, lm in enumerate(handLms.landmark):
                    height, width, channel = img.shape
                    x_pixels, y_pixels = int(lm.x * width), int(lm.y * height)
                    if n in handID:
                        lmInfo.append(lmData(n, id, x_pixels, y_pixels))
                    if draw and id in lmID_draw and n in handID:
                        cv2.circle(img, (x_pixels, y_pixels), drawSize, (0, 255, 0), cv2.FILLED)
        return img, lmInfo


def main():
    cap = cv2.VideoCapture(1)
    detector = HandDetector()

    prevTime = 0
    currentTime = 0

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        img, lmList = detector.findPositions(img, handID=[0, 1], lmID_draw=[0, 4])

        if len(lmList) != 0:
            print(lmList[4])
        currentTime = time.time()

        fps = 1 / (currentTime - prevTime)
        prevTime = currentTime

        cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Video", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()