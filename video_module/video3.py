# Import 'kivy.core.text' must be called in entry point script
# before import of cv2 to initialize Kivy's text provider.
# This fixes crash on app exit.

import kivy.core.text
import cv2
import numpy as np
from kivy.app import App
from kivy.base import EventLoop
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.core.window import Window


class KivyCamera(Image):

    def __init__(self, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = None

    def start(self, capture, fps=30):
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / fps)

    def start2(self, capture, fps=30):
        self.capture = capture
        Clock.schedule_interval(self.update2, 1.0 / fps)

    def stop(self):
        Clock.unschedule(self.update2)
        self.capture = None

    def stop2(self):
        Clock.unschedule(self.update2)
        self.capture = None

    def update(self, dt):
        return_value, frame = self.capture.read()
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (300, 100), (600, 400), (0, 255, 0), 0)
        if return_value:
            texture = self.texture
            w, h = frame.shape[1], frame.shape[0]
            if not texture or texture.width != w or texture.height != h:
                self.texture = texture = Texture.create(size=(w, h))
                texture.flip_vertical()
            texture.blit_buffer(frame.tobytes(), colorfmt='bgr')
            self.canvas.ask_update()

    def update2(self, dt):
        return_value, frame = self.capture.read()
        success2, buffer = cv2.imencode('.jpg', frame)
        buffer.flatten()
        frame = cv2.flip(frame, 1)
        roi = frame[100:400, 300:600]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (31, 31), 0)

        frame = cv2.resize(blur, (128, 128))
        # frame =frame.reshape(128,128,1)
        if return_value:
            texture = self.texture
            w, h = frame.shape[1], frame.shape[0]
            #print('w ='+str(w))
            #print('h = '+str(h))
            if not texture or texture.width != w or texture.height != h:
                self.texture = texture = Texture.create(size=(w, h))
                texture.flip_vertical()
            texture.blit_buffer(frame.tobytes(), colorfmt='luminance')
            self.canvas.ask_update()


capture = None


class QrtestHome(BoxLayout):

    def init_qrtest(self):
        pass

    def dostart(self, *largs):
        global capture
        capture = cv2.VideoCapture(0)
        self.ids.qrcam.start(capture)
        self.ids.qrcam1.start2(capture)

    def dostop(self):
        global capture
        if capture !=None:
            capture.release()
            capture = None
        self.ids.qrcam.stop()
        self.ids.qrcam1.stop2()

    def doexit(self):
        global capture
        if capture != None:
            capture.release()
            capture = None
        EventLoop.close()

    def slider_chanfe(self, value):
        print("Slider value " + str(value))


class qrtestApp(App):

    def build(self):
        Window.clearcolor = (.4,.4,.4,1)
        Window.size = (400, 400)
        homeWin = QrtestHome()
        homeWin.init_qrtest()
        return homeWin

    def on_stop(self):
        global capture
        if capture:
            capture.release()
            capture = None

qrtestApp().run()