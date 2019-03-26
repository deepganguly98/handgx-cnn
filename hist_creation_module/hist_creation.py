import kivy

kivy.require('1.10.1')

import cv2
import numpy as np

from create_histogram import *

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
import pickle
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.factory import Factory
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup

import os

flip = 0

class KivyCamera(Image):

    # init function to initialize the capture variable
    def __init__(self, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        global capture
        capture = cv2.VideoCapture(0)
        self.start(capture)

    # start function for the first color video to to copy capture from the cv2 module and refresh it at regular intervals using 'update' function

    def start(self, capture, fps=30):
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / fps)

    def update(self, dt):
        global roi
        return_value, frame = self.capture.read()
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (300, 100), (600, 400), (0, 255, 0), 0)
        roi = frame[100:400, 300:600]
        roi = draw_rect(roi)
        if return_value:
            texture = self.texture
            w, h = frame.shape[1], frame.shape[0]
            if not texture or texture.width != w or texture.height != h:
                self.texture = texture = Texture.create(size=(w, h))
                texture.flip_vertical()
            texture.blit_buffer(frame.tobytes(), colorfmt='bgr')
            self.canvas.ask_update()

# variable to hold the live video frames
capture = None
roi = None
hand_hist = None

class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class SaveDialog(FloatLayout):
    save = ObjectProperty(None)
    text_input = ObjectProperty(None)
    cancel = ObjectProperty(None)

class HistCreationApp(BoxLayout):
    orient = ObjectProperty(None)
    loadfile = ObjectProperty(None)
    savefile = ObjectProperty(None)

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def show_save(self):
        content = SaveDialog(save=self.save, cancel=self.dismiss_popup)
        self._popup = Popup(title="Save file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def load(self, path, filename):

        with open(os.path.join(path, filename[0]), "rb") as f:
            hist = pickle.load(f)
        #with open(os.path.join(path, filename[0])) as stream:
        #    hist = stream.read()
        print(hist)
        self.dismiss_popup()

    def save(self, path, filename):

        with open(os.path.join(path, filename), "wb") as f:
            pickle.dump(hand_hist, f)

        #with open(os.path.join(path, filename), 'w') as stream:
        #    stream.write(self.text_input.text)

        self.dismiss_popup()

    def __init__(self, **kwargs):
        super(HistCreationApp, self).__init__(**kwargs)
        Window.size = (1350,620)

    def build(self):
        global capture
        capture = cv2.VideoCapture(0)
        self.ids.qrcam.start(capture)

    def generate(self):
        global roi, hand_hist
        hand_hist = hand_histogram(roi)
        print("Histogram generated!")
        print(hand_hist)
        self.show_save()

    def accept(self):
        pass

    def flip(self, val):
        global flip
        if val == 0:
            flip = 1
            #Flip frame
            self.orient.text = 'Left'
            self.orient.color = ( 0.45, 0.95, 0.25, 1)
        else:
            flip = 0
            self.orient.text = 'Right'
            self.orient.color = ( 0.24, 0.64, 0.93, 1)
            #Do no flip frame    

class HistApp(App):
    def build(self):
        return HistCreationApp()


kv = Builder.load_file("hist_creation.kv")

if __name__ == '__main__':
    app = HistApp()
    app.run()
