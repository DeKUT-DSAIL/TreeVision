from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.modalview import ModalView
from kivy.lang import Builder
from kivy.utils import platform
import webbrowser

KV='''
BoxLayout:
    orientation: 'vertical'
    LinkLabel:
        text: "[color=FFFFFF]TreeVision is a software tool that uses 3D computer vision to estimate biophysical parameters to trees. Visit the [/color] [ref=https://github.com/DeKUT-DSAIL/TreeVision][color=BF2700][u]GitHub Repository[/u][/color][/ref] [color=FFFFFF]to learn more[/color]"
        markup: True
'''

class LinkLabel(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(on_ref_press=self.on_link_click)

    def on_link_click(self, instance, value):
        # Open the link in a web browser when clicked
        webbrowser.open(value)

class MyClickableLinkApp(App):
    def build(self):
        return Builder.load_string(KV)

if __name__ == '__main__':
    MyClickableLinkApp().run()
