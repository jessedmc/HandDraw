import cv2 as cv
import mediapipe as mp
import time

print("0000      Begin")
#mpHands = mp.solutions.hands
#hands = mpHands.Hands()
#mpDraw = mp.solutions.drawing_utils
'''
Hands()
def __init__(self,
               static_image_mode=False,
               max_num_hands=2,
               model_complexity=1,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
'''

class HandDetector():
    def __init__(self, mode = False, maxHands = 2, detectionConf = 0.5, trackConf = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackConf = trackConf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, 1, self.detectionConf, self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils
        self.listTipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw = True):
        imgRgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRgb)

        #print("2222    results.multi_hand_landmarks ", self.results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                #print("2222    handLms ", handLms)
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                    #print("2222    type(handLms.landmark) ", type(handLms.landmark))
                    #print("2222    type(enum(handLms.landmark)) ", type(enumerate(handLms.landmark)))
        return img

              #  for idd, lm in enumerate(handLms.landmark):
                  #  print("3333    lm ", lm)
               #     h, w, c = img.shape
               #     cx, cy = int(lm.x * w), int(lm.y * h)
                  #  print("3333   id cx, cy ", idd, cx, cy)
               #     if idd == handPoint:
               #         cv.circle(img, (cx, cy), 5, (0, 255, 0), cv.FILLED)

    def findPosition(self, img, handNum = 0, draw = True):

        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNum]
            for idd, lm in enumerate(myHand.landmark):
                #print("3333    lm ", lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print("3333   id cx, cy ", idd, cx, cy)
                self.lmList.append([idd, cx ,cy])
                if draw:
                    cv.circle(img, (cx, cy), 5, (0, 255, 0), cv.FILLED)
        return self.lmList

    def fingersUp(self):
        listFingers = []

        # thumb
        if self.lmList[self.listTipIds[0]][1] < self.lmList[self.listTipIds[0] - 1][1]:
            listFingers.append(1)
        else:
            listFingers.append(0)

        # 4 fingers
        for idd in range(1, 5):
            if self.lmList[self.listTipIds[idd]][2] < self.lmList[self.listTipIds[idd] - 2][2]:
                listFingers.append(1)
            else:
                listFingers.append(0)

        return listFingers


def main():
    prevTime = 0
    currTime = 0

    #handPoint = 8
    cap = cv.VideoCapture(0)

    cap.set(3, 1920)
    cap.set(4, 1080)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        #print("6666      lmList ", lmList)
        #if len(lmList) > 0:
           # print("6666     lmLIst[8] ", lmList[8])
        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime
        #print("66666          fps ", fps)

        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv.imshow("Live", img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()


