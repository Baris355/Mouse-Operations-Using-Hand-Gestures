import cv2
import mediapipe as mp
import time
import math
import numpy as np


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode # self -> object has its own variable. Assigning value provided by the user. When mode is called program calls value of mode.
        self.maxHands = maxHands
        self.detectionCon = detectionCon  # selfler belirli bir sınıfın belirli bir örneğini(nesnesini) temsil etmek için kullanılır.
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands #bu modeli kullanmadan önce bunu belirlememiz gerekiyor.

        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)   # false -> detects and if confidence level is good then tracks
                                                                            # so it will be faster. Confidencelar %50 nin altına düserse tekrar.
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img: object, draw: object = True) -> object:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # hands objecti yalnızca rgb kullanır. Bu yüzden convert ettik.
        self.results = self.hands.process(imgRGB)      # processes frames and gives results.
        # print(results.multi_hand_landmarks)    # el detect edildi mi edilmedi mi?

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:   #we will have each hand and we will get the info or extract the info for each hand.
                if draw: #Elin belirli yerlerine küçük kırmızı noktaları koyuyor(21 nokta), (Mediapipe'dan alıyor verileri)
                    self.mpDraw.draw_landmarks(img, handLms,  # draw elin üstündeki noktaları çizer.
                                               self.mpHands.HAND_CONNECTIONS) # Noktalar arasındaki bağlantıları çiziyor.

        return img

    def findPosition(self, img, handNo=0, draw=True): # handNo -> whichever hand we want we can ask the info of that. Draw is putted true by default.
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo] # [handNo] -> Belirli bir hand no vermeliyiz. Üstte tanımlandı.
            for id, lm in enumerate(myHand.landmark):  # Bütün landmarkları alıcak. Bir üst satırda bir el için tüm landmarklar belirlendi.
                # print(id, lm)
                h, w, c = img.shape #height, width, coordinates
                cx, cy = int(lm.x * w), int(lm.y * h)  #Elin koordinatlarını öğreniyoruz (x,y , h=height, w=width, c=coordination). Convert dec 2 int.
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED) # 10 = radius

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):
        fingers = []
        # Thumb
        try:
            if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        except:
            pass

        # Fingers
        for id in range(1, 5):
            try:
                if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            except:
                fingers.append(0)

        # totalFingers = fingers.count(1)

        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):   # r-> radius , t-> thickness
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]  # [1:] -> slice.
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED) #mor
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED) #mor
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)   #mavi
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]


    def findDistance2(self, p3, p4, img, draw=True, r=15, t=3):
        x0, y0 = self.lmList[p3][1:]
        x1, y1 = self.lmList[p4][1:]
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2

        if draw:
            cv2.line(img, (x0, y0), (x1, y1), (255, 0, 255), t)
            cv2.circle(img, (x0, y0), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x1 - x0, y1 - y0)

        return length, img, [x0, y0, x1, y1, cx, cy]





def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:

        success, img = cap.read()     #kamera açmak için
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)  # findpositionın içindeki image parametresi önemli !
        if len(lmList) != 0:
            print(lmList[4])

        cv2.imshow("Image", img)   #standart kamera açma işlemleri
        cv2.waitKey(1)

