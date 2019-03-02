import kivy

kivy.require('1.10.0')
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
		
class HomeScreen(Screen):
	pass

class HistScreen(Screen):
	pass

class WorkScreen(Screen):
	pass

class EndScreen(Screen):
	pass

class ManagerScreen(ScreenManager):
	pass

kv = Builder.load_file("App.kv")

class MainApp(App):
	def build(self):
		return kv

if __name__ == '__main__':
	MainApp().run()