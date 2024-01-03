from View.base_screen import BaseScreenView
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.label import MDLabel


class CalibrateScreenView(BaseScreenView):
    def model_is_changed(self) -> None:
        """
        Called whenever any change has occurred in the data model.
        The view in this method tracks these changes and updates the UI
        according to these changes.
        """


class AutoSizedLabel(MDLabel):
    '''
    A label whose size adjusts depending on its contents
    '''
    def on_texture_size(self, instance, value):
        self.height = value[1]


class RefreshConfirm(MDBoxLayout):
    '''
    Popup modal for refreshing the application
    '''
    pass


class InfoPopup(MDBoxLayout):
    '''
    Popup modal that provides information about the DSAIL-TreeVision software
    '''
    pass
