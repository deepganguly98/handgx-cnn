import kivy
kivy.require ('1.10.1')

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.core.window import Window

class HomeScreen(BoxLayout):
	pass

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