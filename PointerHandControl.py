import cv2
import numpy as np
import HandTrackingModule as htm
import time
import pyautogui

# Whiteboard setup
whiteboard = np.ones((600, 800, 3), dtype=np.uint8) * 255  
whiteboardOpen = False 
drawingMode = False  

# Webcam input setup:
wCam, hCam = 640, 480  
frameR = 100 
smoothening = 5  

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.HandDetector(maxHands=1)
wScr, hScr = pyautogui.size()  

while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # 2. Only proceed if landmarks are detected
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  
        x4, y4 = lmList[20][1:]  

        # 3. Check which fingers are up
        fingers = detector.fingersUp()

        # Whiteboard: Open when index and pinky fingers are up (and no other fingers)
        if fingers[1] == 1 and fingers[4] == 1 and sum(fingers) == 2:
            if not whiteboardOpen:
                whiteboardOpen = True  
                print("Whiteboard opened")
            # Show whiteboard open message on the camera feed
            cv2.putText(img, "Whiteboard Open", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # Close whiteboard mode when 5 fingers are up (to close the whiteboard)
        elif whiteboardOpen and sum(fingers) == 5:
            whiteboardOpen = False 
            whiteboard = np.ones((600, 800, 3), dtype=np.uint8) * 255  
            print("Whiteboard closed")
            cv2.putText(img, "Whiteboard Closed", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            time.sleep(0.3)  # Add a slight delay to avoid rapid toggling

        # Drawing on the whiteboard when 2 fingers are up (index and middle)
        elif whiteboardOpen and fingers[1] == 1 and fingers[2] == 1 and sum(fingers) == 2:
            # Convert coordinates to whiteboard size
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, 800))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, 600))

            # Draw on whiteboard
            cv2.circle(whiteboard, (int(x3), int(y3)), 10, (0, 0, 0), cv2.FILLED)
            drawingMode = True

        # 1 Finger Up: Mouse Control Mode (Index finger only)
        elif fingers[1] == 1 and sum(fingers) == 1:
            # Convert Coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # Smoothen the values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # Move Mouse
            pyautogui.moveTo(wScr - clocX, clocY)  # Invert x-axis for accurate control
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # 3 Fingers Up: Left Click (index, middle, and ring fingers)
        elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and sum(fingers) == 3:
            pyautogui.click()  
            time.sleep(0.3)  

        # 4 Fingers Up: Right Click (index, middle, ring, and pinky fingers)
        elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1 and sum(fingers) == 4:
            pyautogui.click(button='right')  
            time.sleep(0.3)  
        # Fist Gesture (0 fingers up): Change Slide
        elif sum(fingers) == 0:
            pyautogui.press('right') 
            print("Slide changed")
            cv2.putText(img, "Slide Changed", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            time.sleep(0.5)  # Delay to avoid rapid slide changes

    # Display whiteboard if it's open
    if whiteboardOpen:
        cv2.imshow("Whiteboard", whiteboard)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
