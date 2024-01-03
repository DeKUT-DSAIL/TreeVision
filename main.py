
"""
Script for managing hot reloading of the project.
For more details see the documentation page -

https://kivymd.readthedocs.io/en/latest/api/kivymd/tools/patterns/create_project/

To run the application in hot boot mode, execute the command in the console:
DEBUG=1 python main.py
"""

import importlib
import os
from kivy import Config
from PIL import ImageGrab

# TODO: You may know an easier way to get the size of a computer display.
resolution = ImageGrab.grab().size

# Change the values of the application window size as you need.
Config.set("graphics", "height", resolution[1])
Config.set("graphics", "width", resolution[0])

from kivy.core.window import Window
from kivy.core.text import LabelBase

# Place the application window on the right side of the computer screen.
Window.top = 30
Window.left = resolution[0] - Window.width

from kivymd.tools.hotreload.app import MDApp
from kivymd.uix.screenmanager import MDScreenManager
from kivymd.font_definitions import theme_font_styles
from kivy.properties import DictProperty


class TreeVision(MDApp):
    KV_DIRS = [os.path.join(os.getcwd(), "View")]
    PREVIOUS_SCREEN = None
    modules = DictProperty()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.manager_screens = MDScreenManager()

    def build_app(self) -> MDScreenManager:
        """
        In this method, you don't need to change anything other than the
        application theme.
        """

        import View.screens
        Window.bind(on_key_down=self.on_keyboard_down)
        importlib.reload(View.screens)
        screens = View.screens.screens

        LabelBase.register(name="Inter", fn_regular="assets/fonts/Inter-Medium.ttf")
        theme_font_styles.append("Inter")
        self.theme_cls.font_styles["Inter"] = ["Inter", 16, False, 0.15]
        self.theme_cls.primary_palette = "Green"
        self.theme_cls.material_style = "M3"
        self.theme_cls.theme_style = "Dark"

        for i, name_screen in enumerate(screens.keys()):
            controller = screens[name_screen]["controller"]()
            view = controller.get_view()
            self.manager_screens = self.manager_screens
            view.name = name_screen
            self.manager_screens.add_widget(view)
        

        self.modules = {
            'Capture': [
                'camera-iris',
                "on_release", lambda x: self.set_screen('main-screen', 'left'),
            ],

            'Calibrate': [
                'cog-outline',
                "on_release", lambda x: self.set_screen('calibrate screen', 'left'),
            ],

            'Extract': [
                'tape-measure',
                "on_release", lambda x: self.set_screen('extract screen', 'right'),
            ]
        }

        return self.manager_screens
    

    def create_image_directories(self):
        images_dir = os.path.join(self.directory, "assets/images/captured")
        thumbnail_dir = os.path.join(self.directory, "assets/images/thumbnails")

        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        elif not os.path.exists(thumbnail_dir):
            os.makedirs(thumbnail_dir)
    

    def on_start(self):
        self.manager_screens.current = 'extract screen'
        self.create_image_directories()
    

    def set_screen(self, name, direction):
        '''
        Sets the current screen to 'name'
        @param name: Name of the screen to switch to
        @param direction: Direction of the transition
        '''
        self.PREVIOUS_SCREEN = self.manager_screens.current
        self.manager_screens.transition.direction = direction
        self.manager_screens.current = name
    

    def on_keyboard_down(self, window, keyboard, keycode, text, modifiers) -> None:
        """
        The method handles keyboard events.

        By default, a forced restart of an application is tied to the
        `CTRL+R` key on Windows OS and `COMMAND+R` on Mac OS.
        """

        if "meta" in modifiers or "ctrl" in modifiers and text == "r":
            self.rebuild()


TreeVision().run()

# After you finish the project, remove the above code and uncomment the below
# code to test the application normally without hot reloading.

# """
# The entry point to the application.
# 
# The application uses the MVC template. Adhering to the principles of clean
# architecture means ensuring that your application is easy to test, maintain,
# and modernize.
# 
# You can read more about this template at the links below:
# 
# https://github.com/HeaTTheatR/LoginAppMVC
# https://en.wikipedia.org/wiki/Model–view–controller
# """
# 


# import os
# from PIL import ImageGrab
# from kivy import Config

# resolution = ImageGrab.grab().size

# Config.set("graphics", "height", resolution[1])
# Config.set("graphics", "width", resolution[0])

# from kivy.core.window import Window

# Window.top = 30
# Window.left = resolution[0] - Window.width

# from kivy.properties import DictProperty
# from kivy.core.text import LabelBase
# from kivymd.font_definitions import theme_font_styles
# from kivymd.app import MDApp
# from kivymd.uix.screenmanager import MDScreenManager

# from View.screens import screens


# class TreeVision(MDApp):
#     KV_DIRS = [os.path.join(os.getcwd(), "View")]
#     PREVIOUS_SCREEN = None
#     modules = DictProperty()
    
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.load_all_kv_files(self.directory)
#         self.manager_screens = MDScreenManager()
        
#     def build(self) -> MDScreenManager:

#         LabelBase.register(name="Inter", fn_regular="assets/fonts/Inter-Medium.ttf")
#         theme_font_styles.append("Inter")
#         self.theme_cls.font_styles["Inter"] = ["Inter", 16, False, 0.15]
#         self.theme_cls.primary_palette = "Green"
#         self.theme_cls.material_style = "M3"
#         self.theme_cls.theme_style = "Dark"

#         self.generate_application_screens()

#         self.modules = {
#             'Capture': [
#                 'camera-iris',
#                 "on_release", lambda x: self.set_screen('main-screen', 'left'),
#             ],

#             'Calibrate': [
#                 'cog-outline',
#                 "on_release", lambda x: self.set_screen('calibrate screen', 'left'),
#             ],

#             'Extract': [
#                 'tape-measure',
#                 "on_release", lambda x: self.set_screen('extract screen', 'right'),
#             ]
#         }
        
#         return self.manager_screens

#     def generate_application_screens(self) -> None:
#         """
#         Creating and adding screens to the screen manager.
#         You should not change this cycle unnecessarily. He is self-sufficient.

#         If you need to add any screen, open the `View.screens.py` module and
#         see how new screens are added according to the given application
#         architecture.
#         """

#         for i, name_screen in enumerate(screens.keys()):
#             controller = screens[name_screen]["controller"]()
#             view = controller.get_view()
#             view.manager_screens = self.manager_screens
#             view.name = name_screen
#             self.manager_screens.add_widget(view)
    

#     def create_image_directories(self):
#         '''
#         Creates some default project directories for storing images
#         '''
#         images_dir = os.path.join(self.directory, "assets/images/captured")
#         thumbnail_dir = os.path.join(self.directory, "assets/images/thumbnails")

#         if not os.path.exists(images_dir):
#             os.makedirs(images_dir)
#         elif not os.path.exists(thumbnail_dir):
#             os.makedirs(thumbnail_dir)
    

#     def on_start(self):
#         '''
#         Fired when the application starts
#         '''
#         self.manager_screens.current = 'extract screen'
#         self.create_image_directories()
    

#     def set_screen(self, name, direction):
#         '''
#         Sets the current screen to 'name'
#         @param name: Name of the screen to switch to
#         @param direction: Direction of the transition
#         '''
#         self.PREVIOUS_SCREEN = self.manager_screens.current
#         self.manager_screens.transition.direction = direction
#         self.manager_screens.current = name


# TreeVision().run()
