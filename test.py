import cv2
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.graphics.texture import Texture

Builder.load_string('''
<CameraWidget>:
    size_hint: 1, 0.9

<MyApp>:
    orientation: 'vertical'

    CameraWidget:
        id: camera_widget

    Button:
        text: 'Exit'
        size_hint: 1, 0.1
        on_release: app.stop()
''')

class CameraWidget(Image):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = cv2.VideoCapture(0)  # Open the default camera (usually 0)
        self.schedule_frame()

    def schedule_frame(self, dt=None):
        ret, frame = self.capture.read()
        if ret:
            self.update_image(frame)
        Clock.schedule_once(self.schedule_frame, 1.0 / 30)  # Schedule the next frame

    def update_image(self, frame):
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize the frame while maintaining aspect ratio
        img_ratio = frame.shape[1] / frame.shape[0]
        widget_ratio = self.width / self.height
        if img_ratio > widget_ratio:
            new_height = int(self.width / img_ratio)
            resized_frame = cv2.resize(frame_rgb, (int(self.width), new_height))
        else:
            new_width = int(self.height * img_ratio)
            resized_frame = cv2.resize(frame_rgb, (new_width, int(self.height)))

        # Convert to Kivy-compatible texture
        buf1 = resized_frame.tostring()
        image_texture = Texture.create(size=(resized_frame.shape[1], resized_frame.shape[0]), colorfmt='rgb')
        image_texture.blit_buffer(buf1, colorfmt='rgb', bufferfmt='ubyte')

        # Update the texture of the Image widget
        self.texture = image_texture

class MyApp(BoxLayout):
    pass

class CameraStreamApp(App):
    def build(self):
        return MyApp()

if __name__ == '__main__':
    CameraStreamApp().run()
