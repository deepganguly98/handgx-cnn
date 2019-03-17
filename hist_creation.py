import kivy
kivy.require ('1.10.1')

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

class KivyCamera(Image):

    #init function to initialize the capture variable
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

#variable to hold the live video frames
capture = None
roi = None

class HistCreationApp(BoxLayout):
    def init_histcreation(self):
        global capture
        capture = cv2.VideoCapture(0)
        self.ids.qrcam.start(capture)

    def generate(self):
    	global roi
    	hand_hist = hand_histogram(roi)
    	print("Histogram generated!")
    	
    def load(self):
    	#Code to load saved histogram and store in hand_hist
    	pass

    def accept(self):
        pass

class HistApp(App):
	def build(self):
		return HistCreationApp()
		
kv = Builder.load_file("hist_creation.kv")

if __name__ == '__main__':
	app = HistApp()
	app.run()