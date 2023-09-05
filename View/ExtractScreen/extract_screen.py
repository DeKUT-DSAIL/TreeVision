from View.base_screen import BaseScreenView
from kivy.properties import ObjectProperty
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.label import MDLabel
from kivymd.app import MDApp
import webbrowser


class ExtractScreenView(BaseScreenView):
    image_plane = ObjectProperty()
    left_im = ObjectProperty()
    right_im = ObjectProperty()
    overlay_layout = ObjectProperty()
    overlay = ObjectProperty()
    next_arrow = ObjectProperty()
    previous_arrow = ObjectProperty()
    project_name = ObjectProperty()
    images_select = ObjectProperty()
    verify_checkbox = ObjectProperty()
    segmentation_dropdown_item = ObjectProperty()
    parameter_dropdown_item = ObjectProperty()


class RefreshConfirm(MDBoxLayout):
    '''
    Popup modal for refreshing the application
    '''
    pass

class InfoPopupModal(MDBoxLayout):
    '''
    Popup modal that provides information about the TreeVision software
    '''
    pass

class LinkLabel(MDLabel):
    '''
    Makes clickable links
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(on_ref_press=self.on_link_click)

    def on_link_click(self, instance, value):
        webbrowser.open(value)