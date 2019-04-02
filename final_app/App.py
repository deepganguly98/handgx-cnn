import kivy

kivy.require('1.10.1')

import numpy as np
import pickle
import pyttsx
import os
import imutils
import cv2
from create_histogram import *

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition

from kivy.core.window import Window

from kivy.uix.label import Label

from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
from kivy.uix.floatlayout import FloatLayout


# _________________
# GLOBAL VARIABLES
# -----------------

flip = 0
# Variable to decide whether to flip the frame depending on hand-orientation chosen.

# variable to hold the live video frames
capture = None

#region of interest
roi = None

#dynamic hsv slider values
u_hue = 104
u_saturation = 135
u_value = 148
l_hue = 0
l_saturation = 0
l_value = 15

#threshold value
t = 128

#variable for concatenation checkbox 
check = True

interval = 3
# From slider

final_mask = None
# To be passed to predict function

flag = 1

timer_val = 3

event = None

# variable to hold the live video frames
capture = None
roi = None
hand_hist = None
hist_name = None

#Splash screen counter
splash_timer = 0

model_alpha = None
model_num = None
model_sym = None

model = model_alpha
model_text = 'Alphabetic model'


class KivyCamera(Image):

    # init function to initialize the capture variable
    def __init__(self, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = None

    # start function for the first color video to to copy capture from the cv2 module and refresh it at regular intervals using 'update' function

    def start(self, capture, fps=30):
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / fps)

    def stop(self):
        Clock.unschedule(self.update)

    def update(self, dt):
        global roi, flip, capture
        return_value, frame = capture.read()
        frame = cv2.flip(frame, 1)
        if flip == 1:
            frame = cv2.flip(frame,1)
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


class KivyCamera2(Image):
    # init function to initialize the capture variable
    def __init__(self, **kwargs):
        super(KivyCamera2, self).__init__(**kwargs)
        self.capture = None
     
    def start(self, capture, fps=30):
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / fps)

    def start1(self, capture, fps=30):
        self.capture = capture
        Clock.schedule_interval(self.update1, 1.0 / fps)

    def stop(self):
        Clock.unschedule(self.update)

    def stop1(self):
        Clock.unschedule(self.update1)

    def manage_image_opr(self, frame):
        global u_hue, u_saturation, u_value, l_hue, l_saturation, l_value, t, hand_hist
        roi = frame[100:400, 300:600]

        cv2.rectangle(frame, (300, 100), (600, 400), (0, 255, 0), 3)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (31, 31), 0)

        try:
            _, thresh = cv2.threshold(blur, t, 255, cv2.THRESH_BINARY_INV)
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            conts = max(cnts, key=lambda x: cv2.contourArea(x))
            M = cv2.moments(conts)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            roi_copy = roi.copy()
            cv2.drawContours(roi, [conts], -1, (0, 255, 0), 2)
            leftmost = tuple(conts[conts[:, :, 0].argmin()][0])
            rightmost = tuple(conts[conts[:, :, 0].argmax()][0])
            topmost = tuple(conts[conts[:, :, 1].argmin()][0])
            bottommost = tuple(conts[conts[:, :, 1].argmax()][0])
            cv2.rectangle(roi, (leftmost[0], topmost[1]), (rightmost[0], topmost[1] + cY), (0, 0, 255), 0)
            roi = roi_copy[topmost[1]:topmost[1] + topmost[1] + cY, leftmost[0]:leftmost[0] + rightmost[0]]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            dst = cv2.calcBackProject([hsv], [0, 1], hand_hist, [0, 180, 0, 256], 1)
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
            cv2.filter2D(dst, -1, disc, dst)
            ret, thresh = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY)
            thresh = cv2.merge((thresh, thresh, thresh))
            r = cv2.bitwise_and(roi, thresh)

            hsv = cv2.cvtColor(r, cv2.COLOR_BGR2HSV)
            lower_skin = np.array([l_hue, l_saturation, l_value], dtype=np.uint8)
            upper_skin = np.array([u_hue, u_saturation, u_value], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            if flip == 1:
                mask = cv2.flip(mask, 1)
            return mask
        except:
            pass
            t = 128
            mask = np.zeros((128, 128), np.uint8)
            return mask

    def trackpalm(self, frame):
        global u_hue, u_saturation, u_value, l_hue, l_saturation, l_value, t

        roi = frame[100:400, 300:600]
        cv2.rectangle(frame, (300, 100), (600, 400), (0, 255, 0), 3)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (31, 31), 0)

        try:
            _, thresh = cv2.threshold(blur, t, 255, cv2.THRESH_BINARY_INV)
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            conts = max(cnts, key=lambda x: cv2.contourArea(x))
            M = cv2.moments(conts)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            roi_copy = roi.copy()
            cv2.drawContours(roi, [conts], -1, (0, 255, 0), 2)
            leftmost = tuple(conts[conts[:, :, 0].argmin()][0])
            rightmost = tuple(conts[conts[:, :, 0].argmax()][0])
            topmost = tuple(conts[conts[:, :, 1].argmin()][0])
            bottommost = tuple(conts[conts[:, :, 1].argmax()][0])
            cv2.rectangle(roi, (leftmost[0], topmost[1]), (rightmost[0], topmost[1] + cY), (0, 0, 255), 0)
        except:
            pass
            t = 128

    def update(self, dt):
        global roi, capture
        return_value, frame = capture.read()
        frame = cv2.flip(frame, 1)
        if flip == 1:
            frame = cv2.flip(frame,1)
        self.trackpalm(frame)
        if return_value:
            texture = self.texture
            w, h = frame.shape[1], frame.shape[0]

            if not texture or texture.width != w or texture.height != h:
                self.texture = texture = Texture.create(size=(w, h))
                texture.flip_vertical()
            texture.blit_buffer(frame.tobytes(), colorfmt='bgr')
            self.canvas.ask_update()

    def update1(self, dt):
        global roi, final_mask, capture
        return_value, frame = capture.read()
        frame = cv2.flip(frame, 1)
        if flip == 1:
            frame = cv2.flip(frame,1)
        roi = frame[100:400, 300:600]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (31, 31), 0)
        mask = self.manage_image_opr(frame)
        final_mask = mask
        frame = cv2.resize(mask, (300, 300))
        if flip == 1:
            frame = cv2.flip(frame,1)
        if return_value:
            texture = self.texture
            w, h = frame.shape[1], frame.shape[0]
            if not texture or texture.width != w or texture.height != h:
                self.texture = texture = Texture.create(size=(w, h))
                texture.flip_vertical()
            texture.blit_buffer(frame.tobytes(), colorfmt='luminance')
            self.canvas.ask_update()

class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class SaveDialog(FloatLayout):
    save = ObjectProperty(None)
    text_input = ObjectProperty(None)
    cancel = ObjectProperty(None)

class SplashScreen(Screen):
    pb = ObjectProperty(None)
    def __init__(self, **kwargs):
        super(SplashScreen, self).__init__(**kwargs)
        Window.size = (500, 300)
        Window.borderless = True
        self.timer_start()

    def timer_start(self):
        Clock.schedule_interval(self.update, 0.1 )

    def timeup(self):
        self.manager.current = 'hist'

    def update(self, dt):
        global splash_timer
        splash_timer = splash_timer + 1
        self.ids.pb.value = splash_timer
        # if (splash_timer == 50):
        #     global model_alpha, model_num, model_sym, model
        #     from keras.models import load_model
        #     model_alpha = load_model('../model/extended_atoz_2.h5')
        #     model_num = load_model('../model/extended_0to9_2.h5')
        #     model_sym = load_model('../model/extended_0to9_2.h5')
        #     model = model_alpha
        if (splash_timer == 100):
            Clock.unschedule(self.update)
            global capture
            capture = cv2.VideoCapture(0)
            self.manager.current = 'hist'

class HistCreationScreen(Screen):
    orient = ObjectProperty(None)
    hist_main = ObjectProperty(None)
    loadfile = ObjectProperty(None)
    savefile = ObjectProperty(None)
    hist_selected = ObjectProperty(None)

    def histenter(self):
        Window.size = (1350, 620)
        Window.borderless = False
        global capture
        self.qrcam.start(capture)

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
        global hist_name, hand_hist
        try:
            with open(os.path.join(path, filename[0]), "rb") as f:
                hand_hist = pickle.load(f)
            hist_name = os.path.basename(filename[0])
            self.hist_selected.text = "Loaded Histogram : " + hist_name
        except:
            pop = Popup(title='Incorrect File Format', content=Label(text = "Choose correct histogram file"),
                    size_hint=(None, None), size=(350, 150))
            pop.open()
        self.dismiss_popup()

    def save(self, path, filename):
        global hist_name
        with open(os.path.join(path, filename), "wb") as f:
            pickle.dump(hand_hist, f)
        hist_name = filename
        self.hist_selected.text = "Generated Histogram : "+hist_name
        self.dismiss_popup()

    def generate(self):
        global roi,hand_hist
        hand_hist = hand_histogram(roi)
        self.show_save()


    def flip(self, val):
        global flip
        if val == 0:
            flip = 1
            # Flip frame
            self.orient.text = 'Left'
            self.orient.color = (0.45, 0.95, 0.25, 1)

        else:
            flip = 0
            self.orient.text = 'Right'
            self.orient.color = (0.24, 0.64, 0.93, 1)
            # Do no flip frame

    def accept(self):
        self.qrcam.stop()
        self.manager.current = 'main'

class MainScreen(Screen):
    pause_text = ObjectProperty(None)
    slider_lbl = ObjectProperty(None)
    timer_lbl = ObjectProperty(None)
    predicted_output = ObjectProperty(None)
    model_used = ObjectProperty(None)
    thresh_lbl = ObjectProperty(None)
    sentence = ObjectProperty(None)
    sent_check = ObjectProperty(None)
    loadfile = ObjectProperty(None)
    lbl_hist = ObjectProperty(None)

    slider_main = ObjectProperty(None)
    qrcam2_1 = ObjectProperty(None)
    qrcam2_2 = ObjectProperty(None)

    u_hue_lbl = ObjectProperty(None)
    u_sat_lbl = ObjectProperty(None)
    u_val_lbl = ObjectProperty(None)
    l_hue_lbl = ObjectProperty(None)
    l_sat_lbl = ObjectProperty(None)
    l_val_lbl = ObjectProperty(None)
    
    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)
        global flag
        flag = 0

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self):
        content = LoadDialog(load=self.load_txt, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def show_save_txt(self):
        content = SaveDialog(save=self.save_txt, cancel=self.dismiss_popup)
        self._popup = Popup(title="Save file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()    

    def save_txt(self, path, filename):
        filename = filename + '.txt'
        with open(os.path.join(path, filename), 'w') as stream:
           stream.write(self.sentence.text)
        self.dismiss_popup()

    def load_txt(self, path, filename):
        try:
            with open(os.path.join(path, filename[0])) as stream:
                self.sentence.text = stream.read()
        except:
            pop = Popup(title='Incorrect File Format', content=Label(text = "Choose a (.txt) file"),
                    size_hint=(None, None), size=(200, 150))
            pop.open()
        self.dismiss_popup()

    def on_start(self):
        global hist_name
        if hist_name == None:
            hist_name = ''
        capture = cv2.VideoCapture(0)
        self.qrcam2_1.start(capture)
        self.qrcam2_2.start1(capture)
        self.ids.lbl_hist.text = "Histogram : "+hist_name

    def model_switch(self, x):
        global model, model_text
        if x == 1:
            model = model_num
            model_text = "Numeric Model"

        if x == 2:
            model = model_alpha
            model_text = "Alphabetic model"

        if x == 3:
            model = model_sym
            model_text = "Symbol model"

        return model_text

    def result_map(self, x):
        ans = ''
        if x == 2:
            ans = '+' 
        elif x == 3:
            ans = '-' 
        elif x == 4:
            ans = '*' 
        elif x == 5:
            ans = '/'
        elif x == 6:
            ans = '?' 
        elif x == 7:
            ans = '!'
        elif x == 8:
            ans = '.'
        elif x == 9:
            ans = ','
        elif x == 10:
            ans = ' '
        elif x == 11:
            ans = chr(8)
        return str(ans)

    # def predict_model(self, mask):
    #     mask = cv2.merge((mask, mask, mask))
    #     gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #     img = cv2.resize(gray, (64, 64))
    #     img2 = img.reshape(1, 64, 64, 1)
    #     prediction = model.predict_classes(img2)
    #     predict_prob = model.predict(img2)
    #     prob = predict_prob[0][np.argmax(predict_prob[0])]
    #     # print(predict_prob[0][np.argmax(predict_prob[0])])

    #     return prediction, prob

    # def predict(self):
    #     # Predicting the output
    #     global final_mask, model_text
    #     prediction, prob = self.predict_model(final_mask)
    #     r = 0
    #     result = ''
    #     if prob >= .80:
    #         if model == model_alpha:
    #             if prediction[0] == 26:
    #                 model_text = self.model_switch(1)
    #                 # model switch
    #                 r = 1
    #             elif prediction[0] == 27:
    #                 model_text = self.model_switch(3)
    #                 # model switch
    #                 r = 1
    #             else:
    #                 if r == 1:
    #                     result = ''
    #                     r = 0
    #                 else:
    #                     result = str(chr(prediction[0] + 65))

    #                 self.predicted_output.text = result + "(prob=" + str(int(prob * 100)) + "%)"
    #                 self.model_used.text = model_text

    #         if model == model_num:
    #             if prediction[0] == 10:
    #                 model_text = self.model_switch(3)
    #                 r = 1
    #             elif prediction[0] == 11:
    #                 model_text = self.model_switch(2)
    #                 r = 1
    #             else:
    #                 if r == 1:
    #                     result = ''
    #                     r = 0
    #                 else:
    #                     result = str(prediction[0])
    #                 self.predicted_output.text = result + "(prob=" + str(int(prob * 100)) + "%)"
    #                 self.model_used.text = model_text

    #         if model == model_sym:
    #             if prediction[0] == 0:
    #                 model_text = self.model_switch(1)
    #                 r = 1
    #             elif prediction[0] == 1:
    #                 model_text = self.model_switch(2)
    #                 r = 1
    #             else:
    #                 if r == 1:
    #                     result = ''
    #                     r = 0
    #                 else:
    #                     result = self.result_map(prediction[0])
    #                 self.predicted_output.text = result + "(prob=" + str(int(prob * 100)) + "%)"
    #                 self.model_used.text = model_text

    #         return result
    def predict(self):
        pass

    def timer_to_predict(self, dt):
        global interval, timer_val, check
        if timer_val > 0:
            self.predict()
            timer_val = timer_val - 1
            if timer_val == 0:
                self.timer_lbl.color = (1, 0, 0, 1)
        else:
            self.timer_lbl.color = (0.65, 0.95, 0.35, 1)
            self.predict()
            timer_val = interval
            result = self.predict()
            if result == None:
                result = ''
            if check == True:
                self.sentence.text = self.sentence.text + result

        self.timer_lbl.text = str(timer_val) + ' s'

    def pause_resume(self):
        global flag, event
        if flag == 0:
            self.pause_text.text = "Pause"
            event = Clock.schedule_interval(self.timer_to_predict, 1)
            flag = 2
        else:
            if self.pause_text.text == "Pause":
                self.pause_text.text = "Resume"
                event.cancel()
            else:
                self.pause_text.text = "Pause"
                event()

    def slider_change_u_hue(self, val):
        global u_hue
        u_hue = int(val)
        self.u_hue_lbl.text = str(u_hue)

    def slider_change_u_saturation(self, val):
        global u_saturation
        u_saturation = int(val)
        self.u_sat_lbl.text = str(u_saturation)

    def slider_change_u_value(self, val):
        global u_value
        u_value = int(val)
        self.u_val_lbl.text = str(u_value)

    def slider_change_l_hue(self, val):
        global l_hue
        l_hue = int(val)
        self.l_hue_lbl.text = str(l_hue)

    def slider_change_l_saturation(self, val):
        global l_saturation
        l_saturation = int(val)
        self.l_sat_lbl.text = str(l_saturation)

    def slider_change_l_value(self, val):
        global l_value
        l_value = int(val)
        self.l_val_lbl.text = str(l_value)

    def thresh_change(self, val):
        global t
        t = int(val)
        self.thresh_lbl.text = str(val)

    def interval_change(self, val):
        global interval, timer_val
        self.slider_lbl.text = str(val) + ' s'
        interval = int(val)
        timer_val = interval
        self.timer_lbl.text = str(timer_val) + ' s'

    def image(self):
        pop = Popup(title='Hand Signs Reference Chart', content=Image(source='../images/ref3.png'),
                    size_hint=(None, None), size=(800, 650))
        pop.open()

    def speak(self):
        engine = pyttsx.init()
        rate = engine.getProperty('rate')
        engine.setProperty('rate', rate - 15)
        if self.sentence.text != '':
            engine.say(self.sentence.text)
        engine.runAndWait()

    def check(self, val):
        global check
        check = val

    def previous(self):
        self.qrcam2_1.stop()
        self.qrcam2_2.stop1()
        self.manager.current = 'hist'

class ScreenManagement(ScreenManager):
    pass

App_kv = Builder.load_file("App.kv")

class MainApp(App):

    def build(self):
        return App_kv

if __name__ == '__main__':
    MainApp().run()