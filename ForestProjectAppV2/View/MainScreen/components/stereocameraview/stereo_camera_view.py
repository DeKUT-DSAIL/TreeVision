from View.base_screen import BaseScreenView
from kivymd.uix.button import MDRectangleFlatIconButton 
from kivymd.uix.behaviors.toggle_behavior import MDToggleButton


class StereoCameraView(BaseScreenView):
    """ Python class for the Two camera view """

class CameraToggleButton(MDRectangleFlatIconButton, MDToggleButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.background_down = self.theme_cls.primary_color 