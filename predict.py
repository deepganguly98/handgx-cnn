# this is to predict in real time
# from preprocessed images and
#pre created histograms
#proy_comment
import cv2
import numpy as np
import math
import imutils
from numpy.core import multiarray
import pickle
from keras.models import load_model
import time

blackscreen = np.zeros((100,100,3), np.uint8)
blackscreen2= np.zeros((100,100,3), np.uint8)

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,50)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

model_alpha = load_model('model/extended_atoz_2.h5')
model_num = load_model('model/extended_0to9_2.h5')

model = model_alpha

cv2.namedWindow('Thresh limit')
cv2.namedWindow('Lower limit')
cv2.namedWindow('Upper limit')

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



cv2.createTrackbar("Thresh", "Thresh limit", 127, 255, l_change_thresh)

cv2.createTrackbar("Hue", "Lower limit",0,255,l_change)
cv2.createTrackbar("Saturation", "Lower limit",20,255,l_change)
cv2.createTrackbar("Value", "Lower limit",70,255,l_change)

cv2.createTrackbar("Hue", "Upper limit",20,255,u_change)
cv2.createTrackbar("Saturation", "Upper limit",255,255,u_change)
cv2.createTrackbar("Value", "Upper limit",255,255,u_change)

#cv2.putText(blackscreen2, "alpha predict", bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
#cv2.imshow("model select",blackscreen2)

def model_switch(x):
    global model
    if x == 1:
        model = model_num
        blackscreen2 = np.zeros((100, 100, 3), np.uint8)
        cv2.putText(blackscreen2, "num predict", bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
        cv2.imshow("model select", blackscreen2)

    if x == 2:
        model = model_alpha
        blackscreen2 = np.zeros((100, 100, 3), np.uint8)
        cv2.putText(blackscreen2, "alpha predict", bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
        cv2.imshow("model select", blackscreen2)



def manage_image_opr(frame):

    roi = frame[100:400, 300:600]

    cv2.rectangle(frame, (300, 100), (600, 400), (0, 255, 0), 0)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (31, 31), 0)

    # l = cv2.getTrackbarPos('Lower', 'Thresh limit')
    # v = cv2.getTrackbarPos('Upper', 'Thresh limit')
    t = cv2.getTrackbarPos('Thresh', 'Thresh limit')

    hue_l = cv2.getTrackbarPos('Hue', 'Lower limit')
    sat_l = cv2.getTrackbarPos('Saturation', 'Lower limit')
    val_l = cv2.getTrackbarPos('Value', 'Lower limit')

    hue_u = cv2.getTrackbarPos('Hue', 'Upper limit')
    sat_u = cv2.getTrackbarPos('Saturation', 'Upper limit')
    val_u = cv2.getTrackbarPos('Value', 'Upper limit')
    #print(t)
    _, thresh = cv2.threshold(blur, t, 255, cv2.THRESH_BINARY_INV)

    #cv2.imshow('contour_thresh', thresh)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(cnts)
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
    cv2.imshow('roi', roi_copy)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    cv2.imshow('hsv',hsv)
    #with open("histogram/hist_home", "rb") as f:
    #    hist = pickle.load(f)
    with open("histogram/hist_home2", "rb") as f:
        hist = pickle.load(f)

    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
    #cv2.imshow('dst', dst)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
    cv2.filter2D(dst, -1, disc, dst)
    ret, thresh = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY)
    thresh = cv2.merge((thresh, thresh, thresh))
    r = cv2.bitwise_and(roi, thresh)
    cv2.imshow("bitwise_and", r)
    hsv = cv2.cvtColor(r, cv2.COLOR_BGR2HSV)

    # lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    # upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    lower_skin = np.array([hue_l, sat_l, val_l], dtype=np.uint8)
    upper_skin = np.array([hue_u, sat_u, val_u], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.merge((mask, mask, mask))
    cv2.imshow('mask', mask)
    cv2.imshow('frame', frame)
    return mask


def predict_model(mask):
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (128, 128))
    cv2.imshow('resized', img)
    img = cv2.resize(gray, (64, 64))
    img2 = img.reshape(1, 64, 64, 1)
    prediction = model.predict_classes(img2)
    predict_prob = model.predict(img2)
    prob = predict_prob[0][np.argmax(predict_prob[0])]
    print(predict_prob[0][np.argmax(predict_prob[0])])

    return prediction, prob


def main():
    #global hand_hist

    capture = cv2.VideoCapture(0)

    while capture.isOpened():
        try:
            blackscreen = np.zeros((100, 100, 3), np.uint8)
            pressed_key = cv2.waitKey(1)
            _, frame = capture.read()
            frame = cv2.flip(frame, 1)
            
            processed = manage_image_opr(frame)
            #processed = manage_image_opr(frame)

            # Predicting the output
            prediction, prob = predict_model(processed)

            #if result_map(str(prediction)) == 26:
            #    model_switch(1)
            #if result_map(str(prediction)) == 27:
            #    model_switch(2)
            #else :
            if prob >=.80:
                if model == model_alpha:
                    if prediction[0] == 26:
                        model_switch(1)
                    if prediction[0] == 27:
                        model_switch(2)
                    else:
                        result = str(chr(prediction[0] + 65))
                        #result = str(chr(result_map(str(prediction)) + 65))
                        cv2.putText(blackscreen, result, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
                        cv2.imshow('Output', blackscreen)
                        print(prediction[0])


                if model == model_num:
                    if prediction[0] == 10:
                        model_switch(1)
                    if prediction[0] == 11:
                        model_switch(2)

                    else:
                        result = str(prediction[0])
                        #result = str(result_map2(str(prediction)))
                        cv2.putText(blackscreen, result, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
                        cv2.imshow('Output', blackscreen)
                        print(prediction[0])

            cv2.imshow("Live Feed", frame)

        except:
            pass
            cv2.imshow('Output', blackscreen)
            cv2.imshow("Live Feed", frame)
        if pressed_key == 27:
            break
    cv2.destroyAllWindows()
    capture.release()


if __name__ == '__main__':
    main()


