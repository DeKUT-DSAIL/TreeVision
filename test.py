from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window
from kivy.app import runTouchApp

layout = GridLayout(cols=1, spacing=10, size_hint_y=None)
# Make sure the height is such that there is something to scroll.
layout.bind(minimum_height=layout.setter('height'))
for i in range(100):
    label = Label(text="Hello World", size_hint_y=None, height=10)
    layout.add_widget(label)

root = ScrollView(bar_width=12)
root.add_widget(layout)

runTouchApp(root)
