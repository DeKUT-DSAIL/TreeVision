from kivy.properties import ObjectProperty, StringProperty

from View.base_screen import BaseScreenView


class MainScreenView(BaseScreenView):
    current_screen = StringProperty()
