#this is to take and store original pictures using a pre-existing histogram
#while seeing how the
import cv2
import numpy as np
import math
import imutils
import pickle


def l_change_thresh(x):
  t = cv2.getTrackbarPos('Lower','Thresh limit')
  #v = cv2.getTrackbarPos('Upper','Thresh limit')


def l_change(x):
  hue_l = cv2.getTrackbarPos('Hue','Lower limit')
  sat_l = cv2.getTrackbarPos('Saturation','Lower limit')
  val_l = cv2.getTrackbarPos('Value','Lower limit')


def u_change(x):
  hue_u = cv2.getTrackbarPos('Hue','Upper limit')
  sat_u = cv2.getTrackbarPos('Saturation','Upper limit')
  val_u = cv2.getTrackbarPos('Value','Upper limit')


cap = cv2.VideoCapture(0)
j = 0
cv2.namedWindow('Thresh limit')


cv2.namedWindow('Lower limit')
cv2.namedWindow('Upper limit')

cv2.createTrackbar("Thresh", "Thresh limit", 0, 255, l_change_thresh)
#cv2.createTrackbar("Upper", "Thresh limit", 255, 255, l_change_thresh)


cv2.createTrackbar("Hue", "Lower limit",0,255,l_change)
cv2.createTrackbar("Saturation", "Lower limit",20,255,l_change)
cv2.createTrackbar("Value", "Lower limit",70,255,l_change)

cv2.createTrackbar("Hue", "Upper limit",20,255,u_change)
cv2.createTrackbar("Saturation", "Upper limit",255,255,u_change)
cv2.createTrackbar("Value", "Upper limit",255,255,u_change)

while (1):
    try:

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame1 = frame.copy()
        # define region of interest
        roi = frame[100:400, 300:600]

        cv2.rectangle(frame, (300, 100), (600, 400), (0, 255, 0), 0)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (31, 31), 0)

        #l = cv2.getTrackbarPos('Lower', 'Thresh limit')
        #v = cv2.getTrackbarPos('Upper', 'Thresh limit')
        t = cv2.getTrackbarPos('Thresh', 'Thresh limit')

        hue_l = cv2.getTrackbarPos('Hue', 'Lower limit')
        sat_l = cv2.getTrackbarPos('Saturation', 'Lower limit')
        val_l = cv2.getTrackbarPos('Value', 'Lower limit')

        hue_u = cv2.getTrackbarPos('Hue', 'Upper limit')
        sat_u = cv2.getTrackbarPos('Saturation', 'Upper limit')
        val_u = cv2.getTrackbarPos('Value', 'Upper limit')
        print(t)
        _, thresh = cv2.threshold(blur, t, 255, cv2.THRESH_BINARY_INV)

        cv2.imshow('contour_thresh',thresh)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        conts = max(cnts, key=lambda x: cv2.contourArea(x))
        M = cv2.moments(conts)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        roi_copy = roi.copy()
        cv2.drawContours(roi, [conts], -1, (0, 255, 0), 2)
        leftmost = tuple(conts[conts[:, :, 0].argmin()][0])
        rightmost = tuple(conts[conts[:, :, 0].argmax()][0])
        topmost = tuple(conts[conts[:, :, 1].argmin()][0])
        bottommost = tuple(conts[conts[:, :, 1].argmax()][0])
        cv2.rectangle(roi, (leftmost[0], topmost[1]), (rightmost[0], topmost[1] + cY), (0, 0, 255), 0)
        roi = roi_copy[topmost[1]:topmost[1] + topmost[1] + cY, leftmost[0]:leftmost[0] + rightmost[0]]
        flip3= cv2.flip(roi, 1)
        cv2.imshow('roi',roi_copy)
        cv2.imshow('Flip_roi', flip3)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        with open("histogram/hist_home", "rb") as f:
            hist = pickle.load(f)
        dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
        cv2.filter2D(dst, -1, disc, dst)
        ret, thresh = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY)
        thresh = cv2.merge((thresh, thresh, thresh))
        r = cv2.bitwise_and(roi, thresh)
        cv2.imshow("bitwise_and", r)
        hsv = cv2.cvtColor(r, cv2.COLOR_BGR2HSV)

        #lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        #upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        lower_skin = np.array([hue_l, sat_l, val_l], dtype=np.uint8)
        upper_skin = np.array([hue_u, sat_u, val_u], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.merge((mask, mask, mask))
        cv2.imshow('mask', mask)
        cv2.imshow('frame', frame)

    except:
        pass
        cv2.imshow('contour_thresh',thresh)
        cv2.imshow('frame', frame)

    k = cv2.waitKey(5) & 0xFF
    # print(k)
    if k == 27:
        break
    if k == 97:
        cv2.imwrite("original_images/s/s" + str(j) + ".jpg", frame1)
        print(j)
        j = j + 1
        if j > 1600:
            break

cv2.destroyAllWindows()
cap.release()