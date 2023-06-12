import cv2 as cv
import os
import HandTrackModule as htm
import numpy as np

# brush thickness
brushThick = 15

# get images for color selection
colorDir = "PaintItColors\\"
imgColorListFile = os.listdir(colorDir)
listColorImg = []
for imgColorFile in imgColorListFile:
    image = cv.imread(colorDir + imgColorFile)
    listColorImg.append(image)

# webcam setup
cap = cv.VideoCapture(0)
cap.set(3, 1920)  # width
cap.set(4, 1080)  # height

detector = htm.HandDetector(detectionConf=0.85)
listFingers = []     # for detection, works with HandMap.png
imgHeader = listColorImg[2] # color selection as green
drawColor = (0, 255, 0)     # default drawing color
xp, yp = 0, 0       # previous drawing positions
imgCanvas = np.zeros((1080, 1920, 3), np.uint8)
while True:
    success, img = cap.read()
    # flip image horiz
    img = cv.flip(img, 1)

    # place color selection image
    img[0:125,0:1920] = imgHeader

    # find hand landmarks
    img = detector.findHands(img, draw = False)
    lmList = detector.findPosition(img, draw = False)

    if len(lmList) > 0:
        # tip of index finger, refer to HandMap.png
        x1, y1 = lmList[8][1:]
        # tip of middle finger
        x2, y2 = lmList[12][1:]

        # check which fingers are up  1 finger for drawing 2 for selection
        listFingers = detector.fingersUp()

        # if selection mode (two fingers) then select not draw
        if listFingers[1] and listFingers[2]:
            xp, yp = 0, 0
            # rectangle for not draw
            cv.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv.FILLED)
            # in the header image, positions for color selection
            if y1 < 125:
                if 412 < x1 < 546:   #   R
                    imgHeader = listColorImg[4]
                    drawColor = (0, 0, 255)
                elif 652 < x1 < 790:   # B
                    imgHeader = listColorImg[0]
                    drawColor = (255, 0, 0)
                elif 885 < x1 < 1018:  # G
                    imgHeader = listColorImg[2]
                    drawColor = (0, 255, 0)
                elif 1114 < x1 < 1263:  # P
                    imgHeader = listColorImg[3]
                    drawColor = (255, 0, 255)
                elif 1350 < x1 < 1488:  # Erase
                    imgHeader = listColorImg[1]
                    drawColor = (0, 0, 0)
        # in drawing mode
        elif listFingers[1] and listFingers[2] == False:
            cv.circle(img, (x1, y1), 15, drawColor, cv.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                brushThick = 100
            else:
                brushThick = 15
            cv.line(img, (xp, yp), (x1, y1), drawColor, brushThick)
            cv.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThick)

            xp, yp = x1, y1

    # image manipulation for better quality drawing
    imgGray = cv.cvtColor(imgCanvas, cv.COLOR_BGR2GRAY)
    _, imgInv = cv.threshold(imgGray, 50, 255, cv.THRESH_BINARY_INV)
    imgInv = cv.cvtColor(imgInv, cv.COLOR_GRAY2BGR)
    img = cv.bitwise_and(img, imgInv)
    imgShow = cv.bitwise_or(img, imgCanvas)

    # if drawing mode (index finger is up)
    #imgShow = cv.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    #cv.imshow("Live", img)
    cv.imshow("Canvas", imgShow)

    cv.waitKey(1)

