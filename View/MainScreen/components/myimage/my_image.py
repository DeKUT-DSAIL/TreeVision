from kivy.properties import ListProperty, StringProperty
from kivy.uix.button import Button
from kivy.uix.image import Image


class MyImage(Image):
    cam = StringProperty()
    image_radius = ListProperty(defaultvalue=[1])
