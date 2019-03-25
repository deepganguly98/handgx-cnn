import kivy

kivy.require('1.10.1')

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from kivy.uix.boxlayout import BoxLayout
from kivy.core.window import Window
import cv2
import numpy as np
from create_histogram import *
import kivy.core.text
from kivy.uix.label import Label
from kivy.base import EventLoop
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup

# _________________
# GLOBAL VARIABLES
# -----------------

flag = 0
flip = 0
# Variable to decide whether to flip the frame depending on hand-orientation chosen.

# variable to hold the live video frames
capture = None
roi = None

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
        cv2.rectangle(frame, (300, 100), (600, 400), (0, 255, 0), 2)
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

class HomeScreen(Screen):
    def __init__(self, **kwargs):
        super(HomeScreen, self).__init__(**kwargs)
        Window.size = (400, 200)

    def flip(self, val):
        global flip
        if val == 0:
            flip = 1
        # Flip frame
        else:
            flip = 0
    # Do no flip frame


class HistCreationScreen(Screen):
	lbl_generate = ObjectProperty(None)
	qrcam1 = ObjectProperty(None)

	def __init__(self, **kwargs):
		super(HistCreationScreen, self).__init__(**kwargs)
		global capture
    	capture = cv2.VideoCapture(0)
    	self.qrcam1.start(capture)

	def generate(self):
		global roi, flag
		hand_hist = hand_histogram(roi)
		if flag == 0:
			self.lbl_generate.text = "Re-take histogram"
		else:
			flag = 2
		self.image()
		print("Histogram generated!")

	def image(self):
		pop = Popup(title='Histogram capture confirmation',
		content=Label(text='Histogram Captured!'),
		size_hint=(None, None), 
		size=(300, 100))
		pop.open()
    	
	def load(self):
    	#Code to load saved histogram and store in hand_hist
		pass

	def accept(self):
		pass

class MainScreen(Screen):
    pass


class OutputScreen(Screen):
    pass


class ScreenManagement(ScreenManager):
    pass


App_kv = Builder.load_file("App.kv")


class MainApp(App):

    def build(self):
        return App_kv


if __name__ == '__main__':
    MainApp().run()
