from View.base_screen import BaseScreenView
from kivy.properties import ObjectProperty


class ExtractScreenView(BaseScreenView):
    camera = ObjectProperty(None)
    verify = ObjectProperty(None)
    left_select = ObjectProperty(None)
    right_select = ObjectProperty(None)
