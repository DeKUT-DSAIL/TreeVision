from kivy.properties import StringProperty
from kivy.uix.button import Button
from kivy.uix.image import Image


class ThumbnailView(Image, Button):
    image_source = StringProperty()
