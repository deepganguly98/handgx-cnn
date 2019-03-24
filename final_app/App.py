# import kivy
# kivy.require ('1.10.1')
#
#
# from kivy.app import App
# from kivy.lang import Builder
# from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
#
# class HomeScreen(Screen):
# 	pass
#
# class HistCreationScreen(Screen):
# 	pass
#
# class MainScreen(Screen):
# 	pass
#
# class OutputScreen(Screen):
# 	pass
#
# class ScreenManagement(ScreenManager):
# 	pass
#
# App_kv = Builder.load_file("App.kv")
#
# class MainApp(App):
#
# 	def build(self):
# 		return App_kv
#
# if __name__ == '__main__':
# 	MainApp().run()

import kivy

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.stacklayout import StackLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.progressbar import ProgressBar
from kivy.properties import ObjectProperty
from kivy.clock import Clock
from kivy.uix.widget import Widget
from kivy.uix.button import Button


# class SplashScreen(Screen):
# 	pass
flip = 0

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


class HistScreen(Screen):
	pass


class WorkScreen(Screen):
	pass


class EndScreen(Screen):
	pass


class ManagerScreen(ScreenManager):
	pass


App_kv = Builder.load_file("App.kv")


class MainApp(App):
	def build(self):
		return App_kv


if __name__ == '__main__':
	MainApp().run()
