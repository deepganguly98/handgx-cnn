import cv2
import numpy as np
import keras
import Tkinter as tk
from matplotlib import pyplot as plt
import imutils


hand_hist = None
traverse_point = []
total_rectangle = 77
hand_rect_one_x = None
hand_rect_one_y = None
hand_rect_two_x = None
hand_rect_two_y = None

def l_change(x):
  hue_l = cv2.getTrackbarPos('Hue','Lower limit')
  sat_l = cv2.getTrackbarPos('Saturation','Lower limit')
  val_l = cv2.getTrackbarPos('Value','Lower limit')
  print('Lower hue = ', hue_l)
  print('Lower saturation = ', sat_l)
  print('Lower value = ', val_l)

def u_change(x):
  hue_u = cv2.getTrackbarPos('Hue','Upper limit')
  sat_u = cv2.getTrackbarPos('Saturation','Upper limit')
  val_u = cv2.getTrackbarPos('Value','Upper limit')
  print('Upper hue = ', hue_u)
  print('Upper saturation = ', sat_u)
  print('Upper value = ', val_u)

def rescale_frame(frame, wpercent=130, hpercent=130):
    width = int(frame.shape[1] * wpercent / 100)
    height = int(frame.shape[0] * hpercent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def contours(hist_mask_image):
    gray_hist_mask_image = cv2.cvtColor(hist_mask_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_hist_mask_image, 0, 255, 0)
    _, cont, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cont


def max_contour(contour_list):
    max_i = 0
    max_area = 0

    for i in range(len(contour_list)):
        cnt = contour_list[i]

        area_cnt = cv2.contourArea(cnt)

        if area_cnt > max_area:
            max_area = area_cnt
            max_i = i

        return contour_list[max_i]

def draw_rect(frame):
    rows, cols, _ = frame.shape
    global total_rectangle, hand_rect_one_x, hand_rect_one_y, hand_rect_two_x, hand_rect_two_y

    k=3
    hand_rect_one_x=np.array(k*rows/20)
    hand_rect_one_y=np.array(11 * cols / 20)
    k=k+1
    hand_rect_one_x=np.append(hand_rect_one_x,[k*rows/20,k*rows/20,k*rows/20])
    hand_rect_one_y=np.append(hand_rect_one_y,[10 * cols / 20, 11 * cols / 20,12 *cols/ 20])
    k=k+1
    hand_rect_one_x = np.append(hand_rect_one_x,[k*rows/20,k*rows/20,k*rows/20,k*rows/20,k*rows/20,k*rows/20])
    hand_rect_one_y = np.append(hand_rect_one_y,[ 14 *cols/ 30, 15 *cols/ 30,16 *cols/ 30,17 *cols/ 30,18 *cols/ 30,19 *cols/ 30])
    k=k+1
    hand_rect_one_x = np.append(hand_rect_one_x,[k*rows/20,k*rows/20,k*rows/20,k*rows/20,k*rows/20])
    hand_rect_one_y = np.append(hand_rect_one_y,[9 * cols / 20, 10 * cols / 20, 11 * cols / 20,12 *cols/ 20, 13* cols/20])
    while(k<=15):
        k=k+1
        if k == 10 or k==9 or k==11:
            hand_rect_one_x = np.append(hand_rect_one_x,[k*rows/20,k*rows/20,k*rows/20,k*rows/20,k*rows/20,k*rows/20,k*rows/20,k*rows/20,k*rows/20])
            hand_rect_one_y = np.append(hand_rect_one_y,[12 * cols/30,13 * cols / 30,14 * cols / 30,15 * cols / 30, 16 * cols / 30, 17 * cols / 30,18 *cols/ 30, 19* cols/30, 20* cols/30])
        else:
            hand_rect_one_x = np.append(hand_rect_one_x,[k*rows/20,k*rows/20,k*rows/20,k*rows/20,k*rows/20])
            hand_rect_one_y = np.append(hand_rect_one_y,[9 * cols / 20, 10 * cols / 20, 11 * cols / 20,12 *cols/ 20, 13* cols/20])


    hand_rect_two_x = hand_rect_one_x + 10
    hand_rect_two_y = hand_rect_one_y + 10

    for i in range(total_rectangle):
        #cv2.rectangle(frame, (hand_rect_one_y[i], hand_rect_one_x[i]), (hand_rect_two_y[i], hand_rect_two_x[i]), (0, 255, 0), 1)
        cv2.rectangle(frame, (int(hand_rect_one_y[i]), int(hand_rect_one_x[i])), (int(hand_rect_two_y[i]), int(hand_rect_two_x[i])), (0, 255, 0), 0)
    return frame



def hand_histogram(frame):
    global hand_rect_one_x, hand_rect_one_y

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = np.zeros([770, 10, 3], dtype=hsv_frame.dtype)

    for i in range(total_rectangle):
        roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[int(hand_rect_one_x[i]):int(hand_rect_one_x[i]) + 10,
                                          int(hand_rect_one_y[i]):int(hand_rect_one_y[i]) + 10]

    hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)

def hist_masking(frame, hist, hue_l, sat_l, val_l, hue_u, sat_u, val_u):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
    #cv2.imshow('roi_hsv',dst )
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    cv2.filter2D(dst, -1, disc, dst)
    #cv2.imshow('roi_filter',dst)
    #blur = cv2.GaussianBlur(dst, (5,5), 0)
    blur2 = cv2.GaussianBlur(gray, (11,11), 0)
    #blur = cv2.medianBlur(blur, 5)
    ret, thresh1 = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)
    ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)
    ret, thesh_otsu = cv2.threshold(blur2, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # thresh = cv2.dilate(thresh, None, iterations=5)
    lower_skin = np.array([hue_l, sat_l, val_l], dtype=np.uint8)
    upper_skin = np.array([hue_u, sat_u, val_u], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    #final = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY)
    mask = cv2.GaussianBlur(mask,(5,5),0)
    mask = cv2.merge((mask,mask,mask))
    cv2.imshow('skin_mask', mask)
    thresh = cv2.merge((thresh, thresh, thresh))
    #thresh = cv2.merge((thresh, thresh, thresh))
    #thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    thresh1 = cv2.merge((thresh1, thresh1, thresh1))
    thresh1 = cv2.cvtColor(thresh1, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('thresh1', thresh1)
    cv2.imshow('thresh', thresh)
    r = cv2.bitwise_and(frame, thresh)
    cv2.imshow('bitwise_and', r)
    #cv2.imshow('thresh2', thesh_otsu)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    conts = max(cnts, key = lambda x: cv2.contourArea(x))
    cv2.drawContours(frame, [conts], -1, (0, 255, 0), 2)
    #return frame
   
    return cv2.bitwise_and(frame, thresh)


def centroid(max_contour):
    moment = cv2.moments(max_contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None


def farthest_point(defects, contour, centroid):
    if defects is not None and centroid is not None:
        s = defects[:, 0][:, 0]
        cx, cy = centroid

        x = np.array(contour[s][:, 0][:, 0], dtype=np.float)
        y = np.array(contour[s][:, 0][:, 1], dtype=np.float)

        xp = cv2.pow(cv2.subtract(x, cx), 2)
        yp = cv2.pow(cv2.subtract(y, cy), 2)
        dist = cv2.sqrt(cv2.add(xp, yp))

        dist_max_i = np.argmax(dist)

        if dist_max_i < len(s):
            farthest_defect = s[dist_max_i]
            farthest_point = tuple(contour[farthest_defect][0])
            return farthest_point
        else:
            return None


def draw_circles(frame, traverse_point):
    if traverse_point is not None:
        for i in range(len(traverse_point)):
            cv2.circle(frame, traverse_point[i], int(5 - (5 * i * 3) / 100), [0, 255, 255], -1)


def manage_image_opr(frame, hand_hist, hue_l, sat_l, val_l, hue_u, sat_u, val_u):
    hist_mask_image = hist_masking(frame, hand_hist, hue_l, sat_l, val_l, hue_u, sat_u, val_u)
    contour_list = contours(hist_mask_image)
    max_cont = max_contour(contour_list)

    cnt_centroid = centroid(max_cont)
    cv2.circle(frame, cnt_centroid, 5, [255, 0, 255], -1)

    if max_cont is not None:
        hull = cv2.convexHull(max_cont, returnPoints=False)
        defects = cv2.convexityDefects(max_cont, hull)
        far_point = farthest_point(defects, max_cont, cnt_centroid)
        print("Centroid : " + str(cnt_centroid) + ", farthest Point : " + str(far_point))
        cv2.circle(frame, far_point, 5, [0, 0, 255], -1)
        if len(traverse_point) < 20:
            traverse_point.append(far_point)
        else:
            traverse_point.pop(0)
            traverse_point.append(far_point)

        draw_circles(frame, traverse_point)


def main():
    global hand_hist
    is_hand_hist_created = False
    capture = cv2.VideoCapture(0)
    l_img = np.zeros((300,512,3), np.uint8)
    u_img = np.zeros((300,512,3), np.uint8)

    while capture.isOpened():
        try:
            pressed_key = cv2.waitKey(1)
            _, frame = capture.read()
            frame = cv2.flip(frame,1)
            roi = frame[100:400,300:600]
            cv2.rectangle(frame, (300,100),(600,400), (0,255,0), 0)

            if pressed_key & 0xFF == ord('z'):
                is_hand_hist_created = True
                hand_hist = hand_histogram(roi)
                
                cv2.namedWindow('Lower limit')
                cv2.namedWindow('Upper limit')

                cv2.createTrackbar("Hue", "Lower limit",0,255,l_change)
                cv2.createTrackbar("Saturation", "Lower limit",37,255,l_change)
                cv2.createTrackbar("Value", "Lower limit",70,255,l_change)

                cv2.createTrackbar("Hue", "Upper limit",55,255,u_change)
                cv2.createTrackbar("Saturation", "Upper limit",255,255,u_change)
                cv2.createTrackbar("Value", "Upper limit",255,255,u_change)


            if is_hand_hist_created:

                cv2.imshow('Lower limit', l_img)
                cv2.imshow('Upper limit', u_img)
           
                hue_l = cv2.getTrackbarPos('Hue','Lower limit')
                sat_l = cv2.getTrackbarPos('Saturation','Lower limit')
                val_l = cv2.getTrackbarPos('Value','Lower limit')

                hue_u = cv2.getTrackbarPos('Hue','Upper limit')
                sat_u = cv2.getTrackbarPos('Saturation','Upper limit')
                val_u = cv2.getTrackbarPos('Value','Upper limit')

                l_img[:] = [val_l, sat_l, hue_l]
                u_img[:] = [val_u, sat_u, hue_u]

                manage_image_opr(roi, hand_hist, hue_l, sat_l, val_l, hue_u, sat_u, val_u)
                
            else:
                roi = draw_rect(roi)

            cv2.imshow("Live Feed", frame)

            if pressed_key == 27:
                break
        except:
            pass
            cv2.imshow("Live Feed", frame)
        if pressed_key == 27:
            break

    cv2.destroyAllWindows()
    capture.release()


if __name__ == '__main__':
    main()
