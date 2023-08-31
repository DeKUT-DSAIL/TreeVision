from kivy.properties import ObjectProperty, StringProperty
from kivymd.app import MDApp
from kivymd.theming import ThemeManager
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.textfield import MDTextField
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton

from View.base_screen import BaseScreenView


class MainScreenView(BaseScreenView):
    current_screen = StringProperty()
    dialog = None
    app = MDApp.get_running_app()
    
    def __init__(self, **kwargs):
        super(MainScreenView, self).__init__(**kwargs)
        self.theme_cls = ThemeManager()
        self.theme_cls.theme_style = "Dark"

    
    def on_enter(self):
        screens = ['calibrate screen', 'extract screen']
        if self.controller.PREVIOUS_SCREEN is None:
            self.show_confirmation_dialog()
    

    def show_confirmation_dialog(self):
        if not self.dialog:
            self.dialog = MDDialog(
                title="Please provide these details:",
                type="custom",
                content_cls=PopupWindow(),
                auto_dismiss = False,
                buttons=[
                    MDFlatButton(
                        text="CONTINUE",
                        theme_text_color="Custom",
                        text_color=self.app.theme_cls.primary_color,
                        on_release=self.controller.accept_dialog_input,
                    )
                ],
            )
        self.dialog.open()


class PopupWindow(MDBoxLayout):
    '''
    Popup window for the user to provide some input
    '''
    pass
