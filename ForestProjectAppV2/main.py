import os
import subprocess
import sys

import cv2
from kivy import Config
from kivy.core.text import LabelBase
from kivy.core.window import Window
from kivymd.app import MDApp
from kivymd.font_definitions import theme_font_styles
from kivymd.uix.screenmanager import MDScreenManager
from kivymd.uix.transition import MDFadeSlideTransition

from View.screens import screens

Config.set('graphics', 'position', 'custom')
Window.size = (800, 480)


class FusionApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.default_camera_index = 0
        self.cameras = []
        self.load_all_kv_files(self.directory)
        self.manager_screens = MDScreenManager()

    def get_camera_indexes(self):
        index = 0
        i = 3
        while i > 0:
            cap = cv2.VideoCapture(index)
            if cap.read()[0]:
                self.cameras.append(index)
                cap.release()
            index += 1
            i -= 1
        return self.cameras

    def get_screen_xy(self):
        output = subprocess.Popen(
            'xrandr | grep "\*" | cut -d" " -f4', shell=True, stdout=subprocess.PIPE
        ).communicate()[0]
        output = str(output).replace("b'", "").replace("\\n'", "")
        self.screenx = int(output.split('x')[0])
        self.screeny = int(output.split('x')[1])

    def center_window(self):
        Window.size = (self.screenx, self.screeny)
        Window.left = 0
        Window.top = 0
        Window.fullscreen = True

    def create_image_directories(self):
        images_dir = os.path.join(self.directory, "assets/images/captured")
        thumbnail_dir = os.path.join(self.directory, "assets/images/thumbnails")

        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        elif not os.path.exists(thumbnail_dir):
            os.makedirs(thumbnail_dir)

    def on_start(self):
        # self.manager_screens.current = 'image-screen'
        self.manager_screens.current_screen.controller.start_stereo_cameras()
        self.create_image_directories()

    def build(self) -> MDScreenManager:
        self.get_camera_indexes()
        self.default_camera_index = self.cameras[0] if self.cameras else print("No Camera attached!")

        LabelBase.register(name="Inter", fn_regular="assets/fonts/Inter-Regular.ttf")
        theme_font_styles.append("Inter")
        self.theme_cls.font_styles['Inter'] = ['Inter', 16, False, 0.15]
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Red"
        self.theme_cls.material_style = "M3"

        self.manager_screens = MDScreenManager(transition=MDFadeSlideTransition())
        self.generate_application_screens()

        if len(sys.argv) > 1:
            self.get_screen_xy()
            self.center_window()

        return self.manager_screens

    def generate_application_screens(self) -> None:
        for i, name_screen in enumerate(screens.keys()):
            controller = screens[name_screen]["controller"]()
            view = controller.get_view()
            view.manager_screens = self.manager_screens
            view.name = name_screen
            self.manager_screens.add_widget(view)


FusionApp().run()
