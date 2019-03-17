import kivy
kivy.require ('1.10.1')

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.core.window import Window

class MainScreen(BoxLayout):
	pass

class App(App):
	def build(self):
		return HomeScreen()
		
kv = Builder.load_file("main.kv")

if __name__ == '__main__':
	app =App()
	app.run()