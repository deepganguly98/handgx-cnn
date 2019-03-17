import kivy
kivy.require ('1.10.1')


from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ObjectProperty
import pyttsx

class SayThis(BoxLayout):
	say_text = ObjectProperty(None)

	def speak(self):
		engine = pyttsx.init()
		rate = engine.getProperty('rate')
		engine.setProperty('rate', rate-15)
		engine.say(self.say_text.text)
		engine.runAndWait()


	def clear(self):
		self.say_text.text = ""
		self.say_text.focus = True

class SayThisApp(App):
	def build(self):
		return SayThis()

kv = Builder.load_file("output.kv")

if __name__ == '__main__':
	app = SayThisApp()
	app.run()