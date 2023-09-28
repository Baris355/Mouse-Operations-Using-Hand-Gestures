import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy
#from autopy.mouse import RIGHT
import pyautogui
from tkinter import *

root = Tk()

# Open window having dimension 100x100
root.geometry('350x350')

# Create a Button


btn1 = Button(root, text='Kamera erişimine izin ver', bd='5', command=root.destroy)

btn2 = Button(root, text= 'Kamera erişimine izin verme', bd='5', command= quit)
# Set the position of button on the top of window.
btn1.pack(side='top')
btn2.pack(side='top')
root.mainloop()

##########################
wCam, hCam = 640, 480
frameR = 100  # Frame Reduction
smoothening = 7
#########################

pTime = 0
plocX, plocY = 0, 0  # önceki lokasyonlar
clocX, clocY = 0, 0  # şuanki lokasyonlar

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1) #landmarkları almak için.
wScr, hScr = autopy.screen.size()
# print(wScr, hScr)




while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)  # bbox -> boundingbox. Eli algılayan yeşil kare.
    # 2. Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # x1,y1 işaret parmağının koordinatları
        x2, y2 = lmList[12][1:] # x2,y2 orta parmağın koordinatları
        # print(x1, y1, x2, y2)

    # 3. Check which fingers are up
    fingers = detector.fingersUp()
    # print(fingers)
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), #ortadaki dikdörtgeni oluşturduk.
                  (255, 0, 255), 2)
    # 4. Only Index Finger : Moving Mode      fingers [1] = 8 , fingers[2] = 12.
    if (fingers[1] == 1 and fingers[2] == 0) and (fingers[1] == 1 and fingers[0] == 0):
        # 5. Convert Coordinates
        x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))  #initial base range from 0 to widthscreen.
        y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
        # 6. Smoothen Values
        clocX = plocX + (x3 - plocX) / smoothening
        clocY = plocY + (y3 - plocY) / smoothening

        # 7. Move Mouse
        autopy.mouse.move(wScr - clocX, clocY)
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        plocX, plocY = clocX, clocY

    # 8. Both Index and middle fingers are up : Clicking Mode
    if fingers[1] == 1 and fingers[2] == 1:
        # 9. Find distance between fingers
        length, img, lineInfo = detector.findDistance(8, 12, img) #line info iki parmagın arasındaki mesafe yeterince kısaldıgında yeşil renk.
        #print(length)
        # 10. Click mouse if distance short
        if length < 40:
            cv2.circle(img, (lineInfo[4], lineInfo[5]), # findpositiondan cx,cy den aldı. Index değerleri.
                       15, (0, 255, 0), cv2.FILLED)

            coords = pyautogui.position()
            pyautogui.leftClick(coords)





    if fingers[0] == 1 and fingers[1] == 1:

        # 9. Find distance between fingers
        length, img, lineInfo = detector.findDistance(4, 8, img)
        #print(length)
        # 10. Click mouse if distance short
        if length < 30:
            cv2.circle(img, (lineInfo[4], lineInfo[5]),
                       15, (0, 255, 0), cv2.FILLED)


            coords = pyautogui.position()
            pyautogui.click(coords, button="right")

    #DENEME
    if fingers[2] == 1 and fingers[4] == 1 :
        length, img, lineInfo = detector.findDistance(4, 20, img)

        if length <40:
            cv2.circle(img, (lineInfo[4], lineInfo[5]),
                        15, (0, 255, 0), cv2.FILLED)


    cv2.imshow("Image",img)
    cv2.waitKey(1)






