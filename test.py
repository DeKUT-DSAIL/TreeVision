from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout

from kivymd.app import MDApp
from kivymd.uix.button import MDFlatButton
from kivymd.uix.dialog import MDDialog

KV = '''
<Content>
    orientation: "vertical"
    spacing: "12dp"
    size_hint_y: None
    height: "120dp"

    MDTextField:
        id: city
        hint_text: "City"

    MDTextField:
        id: street
        hint_text: "Street"

'''


class Content(BoxLayout):
    pass


class Example(MDApp):
    dialog = None

    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Orange"
        return Builder.load_string(KV)
    
    def on_start(self):
        self.show_confirmation_dialog()

    def show_confirmation_dialog(self):
        if not self.dialog:
            self.dialog = MDDialog(
                title="Address:",
                type="custom",
                content_cls=Content(),
                auto_dismiss = False,
                buttons=[
                    MDFlatButton(
                        text="CANCEL",
                        theme_text_color="Custom",
                        text_color=self.theme_cls.primary_color,
                    ),
                    MDFlatButton(
                        text="OK",
                        theme_text_color="Custom",
                        text_color=self.theme_cls.primary_color,
                        on_release=self.close_dialog,
                    ),
                ],
            )
        self.dialog.open()
    

    def close_dialog(self, instance):
        city_text = self.dialog.content_cls.ids.city.text
        street_text = self.dialog.content_cls.ids.street.text
        print(f"City: {city_text}")
        print(f"Street: {street_text}")
        self.dialog.dismiss()


Example().run()