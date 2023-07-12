from kivy.properties import StringProperty
from kivymd.uix.floatlayout import MDFloatLayout


class ImageView(MDFloatLayout):
    image_source = StringProperty()
    image_name = StringProperty()
