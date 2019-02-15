# this is to create and save a histogram for a particular hand and
# light condition
import cv2
import numpy as np
import math
import imutils
import pickle

#cap = cv2.VideoCapture(1)
hand_hist = None
hand_rect_one_x = None
hand_rect_one_y = None

hand_rect_two_x = None
hand_rect_two_y = None
total_rectangle = 75

def draw_rect(frame):
    print("draw rect in")
    rows, cols, _ = frame.shape
    global total_rectangle, hand_rect_one_x, hand_rect_one_y, hand_rect_two_x, hand_rect_two_y

    k = 3
    hand_rect_one_x = np.array(k * rows / 20)
    hand_rect_one_y = np.array(11 * cols / 20)
    k = k + 1
    hand_rect_one_x = np.append(hand_rect_one_x, [k * rows / 20, k * rows / 20, k * rows / 20])
    hand_rect_one_y = np.append(hand_rect_one_y, [10 * cols / 20, 11 * cols / 20, 12 * cols / 20])
    k = k + 1
    hand_rect_one_x = np.append(hand_rect_one_x,
                                [k * rows / 20, k * rows / 20, k * rows / 20, k * rows / 20, k * rows / 20,
                                 k * rows / 20])
    hand_rect_one_y = np.append(hand_rect_one_y,
                                [14 * cols / 30, 15 * cols / 30, 16 * cols / 30, 17 * cols / 30, 18 * cols / 30,
                                 19 * cols / 30])
    k = k + 1
    hand_rect_one_x = np.append(hand_rect_one_x,
                                [k * rows / 20, k * rows / 20, k * rows / 20, k * rows / 20, k * rows / 20])
    hand_rect_one_y = np.append(hand_rect_one_y,
                                [9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 12 * cols / 20, 13 * cols / 20])
    while (k <= 14):
        k = k + 1
        if k == 10 or k == 9 or k == 11:
            hand_rect_one_x = np.append(hand_rect_one_x,
                                        [k * rows / 20, k * rows / 20, k * rows / 20, k * rows / 20, k * rows / 20,
                                         k * rows / 20, k * rows / 20, k * rows / 20, k * rows / 20])
            hand_rect_one_y = np.append(hand_rect_one_y,
                                        [12 * cols / 30, 13 * cols / 30, 14 * cols / 30, 15 * cols / 30, 16 * cols / 30,
                                         17 * cols / 30, 18 * cols / 30, 19 * cols / 30, 20 * cols / 30])
        else:
            hand_rect_one_x = np.append(hand_rect_one_x,
                                        [k * rows / 20, k * rows / 20, k * rows / 20, k * rows / 20, k * rows / 20])
            hand_rect_one_y = np.append(hand_rect_one_y,
                                        [9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 12 * cols / 20, 13 * cols / 20])
    k = k + 1
    hand_rect_one_x = np.append(hand_rect_one_x, [k * rows / 20, k * rows / 20, k * rows / 20])
    hand_rect_one_y = np.append(hand_rect_one_y, [10 * cols / 20, 11 * cols / 20, 12 * cols / 20])

    hand_rect_two_x = hand_rect_one_x + 10
    hand_rect_two_y = hand_rect_one_y + 10

    for i in range(total_rectangle):
        cv2.rectangle(frame, (hand_rect_one_y[i], hand_rect_one_x[i]), (hand_rect_two_y[i], hand_rect_two_x[i]),
                      (0, 255, 0), 1)
    return frame


def hand_histogram(frame):
    global hand_rect_one_x, hand_rect_one_y

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = np.zeros([750, 10, 3], dtype=hsv_frame.dtype)

    # print(total_rectangle)
    for i in range(total_rectangle):
        roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[hand_rect_one_x[i]:hand_rect_one_x[i] + 10,
                                          hand_rect_one_y[i]:hand_rect_one_y[i] + 10]

    # print(len(roi))
    hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)


def main():
    global hand_hist, j
    is_hand_hist_created = False
    print('inside')
    capture = cv2.VideoCapture(0)
    while capture.isOpened():
        try:
            pressed_key = cv2.waitKey(1)
            _, frame = capture.read()

            frame = cv2.flip(frame, 1)
            #frame2 = frame.copy()
            roi = frame[100:400, 300:600]
            cv2.rectangle(frame, (300, 100), (600, 400), (0, 255, 0), 0)
            if pressed_key & 0xFF == ord('z'):
                is_hand_hist_created = True
                hand_hist = hand_histogram(roi)
            if is_hand_hist_created:
                print(is_hand_hist_created)
                with open("histogram/hist_deep2", "wb") as f:
                    pickle.dump(hand_hist, f)
                break
            else:
                roi = draw_rect(roi)
                print("draw rect outside")

            cv2.imshow("Live Feed", frame)

        except:
            pass
            cv2.imshow("Live Feed", frame)
        if pressed_key == 27:
            break
    cv2.destroyAllWindows()
    capture.release()


if __name__ == '__main__':
    main()
