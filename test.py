from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.boxlayout import BoxLayout

class MyApp(App):
    def build(self):
        # Create a horizontal box layout
        box = BoxLayout(orientation='horizontal')
        
        # Create four anchor layouts with different anchors
        anchor1 = AnchorLayout(anchor_x='left')
        anchor2 = AnchorLayout(anchor_x='center')
        anchor3 = AnchorLayout(anchor_x='right')
        anchor4 = AnchorLayout(anchor_x='center')
        
        # Add buttons to each anchor layout
        btn1 = Button(text='Column 1')
        btn2 = Button(text='Column 2')
        btn3 = Button(text='Column 3')
        btn4 = Button(text='Column 4')
        
        anchor1.add_widget(btn1)
        anchor2.add_widget(btn2)
        anchor3.add_widget(btn3)
        anchor4.add_widget(btn4)
        
        # Add the anchor layouts to the box layout
        box.add_widget(anchor1)
        box.add_widget(anchor2)
        box.add_widget(anchor3)
        box.add_widget(anchor4)
        
        return box

if __name__ == '__main__':
    MyApp().run()
