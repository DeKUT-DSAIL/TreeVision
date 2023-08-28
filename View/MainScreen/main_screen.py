from kivy.properties import ObjectProperty, StringProperty
from kivymd.theming import ThemeManager
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.textfield import MDTextField
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton

from View.base_screen import BaseScreenView


class MainScreenView(BaseScreenView):
    current_screen = StringProperty()
    
    def __init__(self, **kwargs):
        super(MainScreenView, self).__init__(**kwargs)
        self.theme_cls = ThemeManager()
        self.theme_cls.theme_style = "Dark"
        print(f"W: {self.width}, H: {self.height}")
    

    def show_confirmation_dialog(self):
        self.dialog = MDDialog(
            title="Please provide these details:",
            type="custom",
            content_cls=PopupWindow(),
            buttons=[
                MDFlatButton(
                    text="CONTINUE",
                    theme_text_color="Custom",
                    text_color=self.app.theme_cls.primary_color,
                )
            ],
        )
        self.dialog.open()


class PopupWindow(MDBoxLayout):
    '''
    Popup window for the user to provide some input
    '''
    def __init__(self, **kwargs):
        super(PopupWindow, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.spacing = '12dp'
        self.size_hint_y = None
        self.height = '200dp'

        project_field = MDTextField(id = "project_name", hint_text="Project name")
        frame_width_field = MDTextField(id = "frame_width", hint_text="Frame width")
        frame_height_field = MDTextField(id = "frame_height", hint_text="Frame height")

        self.add_widget(project_field)
        self.add_widget(frame_width_field)
        self.add_widget(frame_height_field)
