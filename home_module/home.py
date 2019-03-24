import kivy
kivy.require ('1.10.1')

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.core.window import Window

flip = 0
#Variable to decide whether to flip the frame depending on hand-orientation chosen.

class HomeScreen(BoxLayout):

	def __init__(self, **kwargs):
		super(HomeScreen, self).__init__(**kwargs)

	def flip(self, val):
		global flip
		if val == 0:
			flip = 1
			#Flip frame
		else:
			flip = 0
			#Do no flip frame

class HomeApp(App):
	def __init__(self, **kwargs):
		super(HomeApp, self).__init__(**kwargs)
		Window.size = (400, 200)
	def build(self):
		return HomeScreen()
		
kv = Builder.load_file("home.kv")

if __name__ == '__main__':
	app = HomeApp()
	app.run()