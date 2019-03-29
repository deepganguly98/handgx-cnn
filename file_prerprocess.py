import cv2, glob
import numpy as np
import math
import imutils
import pickle

with open("histogram/hist_lab2", "rb") as f:
    hist = pickle.load(f)


# count the number of images
def preprocess(iterate, classifier):
    j = 0
    for j in range(classifier):
        print("---" + iterate[j] + "---")
        i = 0
        for img in glob.glob("original_images/" + iterate[j] + "/*.jpg"):
            cv_img = cv2.imread(img)
            print(i)
            kernel = np.ones((3, 3), np.uint8)
            roi = cv_img[100:400, 300:600]
            # canny = cv2.Canny(roi, 60,60)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (31, 31), 0)

            _, thresh = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            conts = max(cnts, key=lambda x: cv2.contourArea(x))

            M = cv2.moments(conts)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # img = np.zeros((300,300,3), np.uint8)
            # cv2.drawContours(img, [conts], -1, (0, 255, 0), 2)
            leftmost = tuple(conts[conts[:, :, 0].argmin()][0])
            rightmost = tuple(conts[conts[:, :, 0].argmax()][0])
            topmost = tuple(conts[conts[:, :, 1].argmin()][0])
            bottommost = tuple(conts[conts[:, :, 1].argmax()][0])
            # img = img[topmost[1]:topmost[1]+topmost[1]+cY, leftmost[0]:leftmost[0]+rightmost[0]]
            roi = roi[topmost[1]:topmost[1] + topmost[1] + cY, leftmost[0]:leftmost[0] + rightmost[0]]
            # img = cv2.resize(img, (64, 64))
            # thresh = cv2.resize(thresh,(64,64))
            # canny = cv2.resize(canny, (64,64))
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
            cv2.filter2D(dst, -1, disc, dst)
            ret, thresh = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY)
            thresh = cv2.merge((thresh, thresh, thresh))
            r = cv2.bitwise_and(roi, thresh)

            hsv = cv2.cvtColor(r, cv2.COLOR_BGR2HSV)
            #lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            #upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            #lower_skin = np.array([0, 35, 70], dtype=np.uint8)
            #upper_skin = np.array([35, 255, 255], dtype=np.uint8)
            
            lower_skin = np.array([0, 15, 70], dtype=np.uint8)
            upper_skin = np.array([35, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            mask = cv2.merge((mask, mask, mask))
            re = cv2.resize(mask, (128, 128))
            
            # cv2.imwrite("orig_canny_processed/"+iterate[j]+"/"+iterate[j]+str(i)+".jpg",canny)
            # cv2.imwrite("orig_contour_processed/"+iterate[j]+"/"+iterate[j]+str(i)+".jpg",img)
            
            cv2.imwrite("orig_final_threshold_processed_lab2/" + iterate[j] + "/" + iterate[j] + str(i) + ".jpg", re)
            i = i + 1


# classifier = 38
classifier = 1
iterate = ["e2"]

# iterate = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","0","1","2","3","4","5","6","7","8","9","e1","e2"]
# iterate = ["a"]
preprocess(iterate, classifier)





