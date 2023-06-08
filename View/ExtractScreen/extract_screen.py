from View.base_screen import BaseScreenView
from kivy.properties import ObjectProperty


class ExtractScreenView(BaseScreenView):
    image_plane = ObjectProperty()
    left_im = ObjectProperty()
    right_im = ObjectProperty()
    project_name = ObjectProperty()
    images_select = ObjectProperty()
    verify_checkbox = ObjectProperty()
    segmentation_dropdown_item = ObjectProperty()
    parameter_dropdown_item = ObjectProperty()
