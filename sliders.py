import kivy
kivy.require ('1.10.1')

import cv2
import numpy as np

import imutils
import pickle
import kivy.core.text
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.core.window import Window
from kivy.base import EventLoop
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.properties import ObjectProperty

u_hue = 15
u_saturation = 15
u_value = 15
l_hue = 15
l_saturation = 15
l_value = 15
t = 128
class KivyCamera(Image):

    #init function to initialize the capture variable
    def __init__(self, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = None

    def start(self, capture, fps=30):
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / fps)

    def start1(self, capture, fps=30):
        self.capture = capture
        Clock.schedule_interval(self.update1, 1.0 / fps)

    def manage_image_opr(self, frame):
        global u_hue,u_saturation,u_value,l_hue,l_saturation,l_value, t
        roi = frame[100:400, 300:600]

        cv2.rectangle(frame, (300, 100), (600, 400), (0, 255, 0), 0)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (31, 31), 0)

        try:
            _, thresh = cv2.threshold(blur, t, 255, cv2.THRESH_BINARY_INV)
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            conts = max(cnts, key=lambda x: cv2.contourArea(x))
        except:
            t=128
            _, thresh = cv2.threshold(blur, t, 255, cv2.THRESH_BINARY_INV)
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
        #cv2.imshow('roi', roi_copy)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        #with open("histogram/hist_home", "rb") as f:
        #    hist = pickle.load(f)
        with open("histogram/hist_home2", "rb") as f:
            hist = pickle.load(f)

        dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
        cv2.filter2D(dst, -1, disc, dst)
        ret, thresh = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY)
        thresh = cv2.merge((thresh, thresh, thresh))
        r = cv2.bitwise_and(roi, thresh)
        #cv2.imshow("bitwise_and", r)
        hsv = cv2.cvtColor(r, cv2.COLOR_BGR2HSV)

        # lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        # upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        lower_skin = np.array([l_hue, l_saturation, l_value], dtype=np.uint8)
        upper_skin = np.array([u_hue, u_saturation, u_value], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        #mask = cv2.merge((mask, mask, mask))
        print(mask.shape)
        #cv2.imshow('mask', mask)
        return mask

    def update(self, dt):
        global roi
        return_value, frame = self.capture.read()
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (300, 100), (600, 400), (0, 255, 0), 0)
        #mask = self.manage_image_opr(frame)

        if return_value:
            texture = self.texture
            w, h = frame.shape[1], frame.shape[0]

            if not texture or texture.width != w or texture.height != h:
                self.texture = texture = Texture.create(size=(w, h))
                texture.flip_vertical()
            texture.blit_buffer(frame.tobytes(), colorfmt='bgr')
            self.canvas.ask_update()
    def update1(self, dt):
        global roi
        return_value, frame = self.capture.read()
        frame = cv2.flip(frame, 1)
        roi = frame[100:400, 300:600]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (31, 31), 0)
        # roi = frame[100:400, 300:600]
        # roi = draw_rect(roi)
        mask = self.manage_image_opr(frame)
        #mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('bur',blur)
        # mask.reshape(128,128)
        #mask = cv2.resize(mask,(128,128))
        frame = cv2.resize(mask, (128, 128))
        if return_value:
            texture = self.texture
            w, h = frame.shape[1], frame.shape[0]
            # w, h = frame.shape[1], frame.shape[0]
            if not texture or texture.width != w or texture.height != h:
                self.texture = texture = Texture.create(size=(w, h))
                texture.flip_vertical()
            texture.blit_buffer(frame.tobytes(), colorfmt='luminance')
            self.canvas.ask_update()





#variable to hold the live video frames
capture = None
roi = None

class HslSliderApp(BoxLayout):
    def __init__(self, **kwargs):
        super(HslSliderApp, self).__init__(**kwargs)
        Window.size = (1000,300)
        capture = cv2.VideoCapture(0)
        self.ids.qrcam.start(capture)
        self.ids.qrcam1.start1(capture)

    def slider_change_u_hue(self, val):
        global u_hue
        u_hue = val 
        #print("u_hue = ",val)

    def slider_change_u_saturation(self, val):
        global u_saturation
        u_saturation = val
        #print("u_saturation = ",val)

    def slider_change_u_value(self, val):
        global u_value
        u_value = val
        #print("u_value = ",val)

    def slider_change_l_hue(self, val):
        global l_hue
        l_hue = val
        #print("l_hue = ",val)

    def slider_change_l_saturation(self, val):
        global l_saturation
        l_saturation = val
        #print("l_saturation = ",val)

    def slider_change_l_value(self, val):
        global l_value
        l_value = val
        #print("l_value = ",val)

    def thresh_change(self,val):
        global t
        t = val


class SliderApp(App):
	def build(self):
		return HslSliderApp()
		
kv = Builder.load_file("sliders.kv")

if __name__ == '__main__':
	app = SliderApp()
	app.run()