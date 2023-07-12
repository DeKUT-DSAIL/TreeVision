from kivy.properties import ObjectProperty, StringProperty
from kivymd.theming import ThemeManager

from View.base_screen import BaseScreenView


class MainScreenView(BaseScreenView):
    current_screen = StringProperty()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.theme_cls = ThemeManager()
        self.theme_cls.theme_style = "Dark"
