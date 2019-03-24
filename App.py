import kivy
kivy.require ('1.10.1')


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
#Variable to decide whether to flip the frame depending on hand-orientation chosen.

#variable to hold the live video frames
capture = None
roi = None

class HomeScreen(Screen):
	def __init__(self, **kwargs):
		super(HomeScreen, self).__init__(**kwargs)
		Window.size = (400, 200)

	def flip(self, val):
		global flip
		if val == 0:
			flip = 1
			#Flip frame
		else:
			flip = 0
			#Do no flip frame

class HistCreationScreen(Screen):
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