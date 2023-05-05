from View.base_screen import BaseScreenView
from kivy.properties import ObjectProperty


class ExtractScreenView(BaseScreenView):
    image_plane = ObjectProperty()
    left_im = ObjectProperty()
    right_im = ObjectProperty()
    verify_checkbox = ObjectProperty()
