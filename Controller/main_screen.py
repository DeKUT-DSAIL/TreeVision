import importlib
import os.path
import time
from functools import partial
from sys import platform

import cv2
from PIL import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.properties import ObjectProperty, StringProperty
from kivymd.toast import toast
from kivymd.uix.menu import MDDropdownMenu

import View.MainScreen.main_screen
from View.MainScreen import ThumbnailView, CameraMenuHeader

# We have to manually reload the view module in order to apply the
# changes made to the code on a subsequent hot reload.
# If you no longer need a hot reload, you can delete this instruction.
importlib.reload(View.MainScreen.main_screen)


class MainScreenController:
    """
    The `MainScreenController` class represents a controller implementation.
    Coordinates work of the view with the model.
    The controller implements the strategy pattern. The controller connects to
    the view to control its actions.
    """

    prev_cam_id = None
    PREVIOUS_SCREEN = None

    PROJECT_NAME = None
    FRAME_WIDTH = None
    FRAME_HEIGHT = None

    def __init__(self):
        self.left_current_image = None
        self.right_current_image = None
        self.current_image = None
        self.video_event = None
        self.left_video_event = None
        self.right_video_event = None
        self.view = View.MainScreen.main_screen.MainScreenView(controller=self)
        self.height = self.view.ids.camera_screen.ids.camera_canvas.height
        self.width = self.view.ids.camera_screen.ids.camera_canvas.width
        self.stereo_flag = False
        self.single_flag = False
        self.root_folder = os.path.join(os.getcwd(), "assets/images/captured")
        self.root_thumbnail = os.path.join(os.getcwd(), "assets/images/thumbnails")

        self.default_camera_index = 0
        self.cameras = []
        
        self.get_camera_indexes()
        self.default_camera_index = self.cameras[0] if self.cameras else toast("No Camera attached!")

        if len(self.cameras) >= 2:
            self.left_cam_index = self.cameras[0]
            self.right_cam_index = self.cameras[1]
            self.view.ids.stereo_tab.disabled = False
        
        elif len(self.cameras) == 1:
            self.view.ids.stereo_tab.disabled = True

        camera_items = [
            {
                "text": f"Camera {i+1}",
                "viewclass": "OneLineListItem",
                "font_style": "Inter",
                "on_release": lambda x=i: self.switch_camera(x),
            } for i in self.cameras[::-1]
        ]

        self.camera_menu_object = MDDropdownMenu(
            background_color=self.view.app.theme_cls.bg_normal,
            header_cls=CameraMenuHeader(),
            caller=self.view.ids.camera_screen.ids.camera_menu,
            items=camera_items,
            width_mult=4,
        )
    

    def get_camera_indexes(self):
        '''
        Returns the indexes of all attached cameras
        '''
        index = 0
        i = 3
        while i > 0:
            cap = cv2.VideoCapture(index)
            if cap.read()[0]:
                self.cameras.append(index)
                cap.release()
            index += 1
            i -= 1
        print(f"CAMS: {self.cameras}")
        return self.cameras


    def close_dialog(self, instance):
        self.view.dialog.dismiss()
    

    def accept_dialog_input(self, instance):
        self.PROJECT_NAME = self.view.dialog.content_cls.ids.project_name.text
        self.FRAME_WIDTH = int(self.view.dialog.content_cls.ids.frame_width.text)
        self.FRAME_HEIGHT = int(self.view.dialog.content_cls.ids.frame_height.text)

        if self.PROJECT_NAME == '':
            toast("Please provide a project name")
        
        else:
            project_path = os.path.join("assets/images/captured", self.PROJECT_NAME)
            if not os.path.exists(project_path):
                os.makedirs(project_path)
            
            self.view.dialog.dismiss()


    def change_default_tab(self, *args):
        '''
        Sets the default tab on the capture screen. There are two tabs i.e., the stereo camera tab and the single camera tab
        '''
        self.view.ids.bottom_navigation.switch_tab("screen 2")
    


    def switch_screen(self, screen_name):
        '''
        Switches to the screen with the name `screen_name`
        @param screen_name: Name of the screen to switch to
        '''
        self.PREVIOUS_SCREEN = self.view.manager_screens.current
        self.view.manager_screens.current = screen_name
        if screen_name == 'main-screen':
            self.close_dialog()
    
    def swap_cameras(self):
        '''
        Swaps the indices of the left and right cameras
        '''
        if self.left_cam_index == self.cameras[-2]:
            self.left_cam_index = self.cameras[-1]
            self.right_cam_index = self.cameras[-2]
        else:
            self.left_cam_index = self.cameras[-2]
            self.right_cam_index = self.cameras[-1]
        self.stop_stereo_cameras()
        self.start_stereo_cameras()


    def capture_image(self, stereo=False):
        """ Function to capture the images """
        time_str = time.strftime("%Y%m%d_%H%M%S")
        if not stereo:
            image_path = f"{self.root_folder}/{self.PROJECT_NAME}/IMG_{time_str}.jpg"
            self.current_image = image_path
            thumbnail_path = f"{self.root_thumbnail}/IMG_{time_str}.jpg"
            thumbnail_widget = ThumbnailView()
            thumbnail_widget.controller = self.view.controller
            if cv2.imwrite(image_path, self.image_frame):
                image = Image.open(image_path)
                image.thumbnail((60, 60))
                image.save(thumbnail_path)
                toast("Image Saved")
                thumbnail_widget.image_source = thumbnail_path
                self.view.ids.camera_screen.ids.thumbnail_section.add_widget(thumbnail_widget)
            else:
                toast("Failed to save")
        else:
            left_image_path = f"{self.root_folder}/{self.PROJECT_NAME}/IMG_{time_str}_LEFT.jpg"
            right_image_path = f"{self.root_folder}/{self.PROJECT_NAME}/IMG_{time_str}_RIGHT.jpg"
            self.right_current_image = right_image_path
            self.left_current_image = left_image_path
            left_thumbnail_path = f"{self.root_thumbnail}/IMG_{time_str}_LEFT.jpg"
            right_thumbnail_path = f"{self.root_thumbnail}/IMG_{time_str}_RIGHT.jpg"

            left_thumbnail_widget = ThumbnailView()
            right_thumbnail_widget = ThumbnailView()

            right_thumbnail_widget.controller = self.view.controller
            left_thumbnail_widget.controller = self.view.controller
            if cv2.imwrite(right_image_path, self.right_image_frame) and cv2.imwrite(left_image_path, self.left_image_frame):
                image = Image.open(right_image_path)
                image.thumbnail((60, 60))
                image.save(right_thumbnail_path)
                right_thumbnail_widget.image_source = right_thumbnail_path

                image = Image.open(left_image_path)
                image.thumbnail((60, 60))
                image.save(left_thumbnail_path)
                left_thumbnail_widget.image_source = left_thumbnail_path

                self.view.ids.stereo_camera_screen.ids.left_thumbnail_section.add_widget(left_thumbnail_widget)
                self.view.ids.stereo_camera_screen.ids.right_thumbnail_section.add_widget(right_thumbnail_widget)

                toast("Images Saved")
            else:
                toast("Failed to save images")

    def switch_camera(self, camera_index):
        '''
        In the single camera screen, this function is used to change the streaming camera
        '''
        self.prev_cam_id = camera_index
        self.stop_camera()
        self.start_camera(cam_id=camera_index)

    def show_image(self, instance):
        '''
        Displays a selected image on the screen
        '''
        image_name = instance.image_source.split("/")[-1]
        image_path = os.path.join(self.root_folder, self.PROJECT_NAME, image_name)
        image_screen = self.view.manager_screens.get_screen("image-screen")
        image_screen.ids.image_section.image_source = image_path
        image_screen.ids.image_section.image_name = image_name
        self.switch_screen('image-screen')

    def load_video(self, side, *args) -> None:
        if side == "left":
            ret, frame = self.left_capture.read()
            self.left_image_frame = frame
            self.left_camera.texture = self.create_texture(frame)
        elif side == "right":
            ret, frame = self.right_capture.read()
            self.right_image_frame = frame
            self.right_camera.texture = self.create_texture(frame)
        elif side == "single":
            ret, frame = self.capture.read()
            self.image_frame = frame
            self.image.texture = self.create_texture(frame)

    def create_texture(self, frame) -> Texture:
        try:
            frame = cv2.flip(frame, 0)
            buffer = frame.tostring()
            image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')

            return image_texture
        
        except AttributeError:
            return Texture.create(size=(self.FRAME_HEIGHT, self.FRAME_WIDTH), colorfmt='bgr')


    def stop_camera(self, *args):
        self.single_flag = False
        self.video_event.cancel()
        self.capture.release()

        texture = Texture.create(size=(self.FRAME_WIDTH, self.FRAME_HEIGHT))
        texture.blit_buffer(bytes([255, 255, 255] * self.FRAME_WIDTH * self.FRAME_HEIGHT), colorfmt='rgb', bufferfmt='ubyte')
        self.image.texture = texture

    def start_camera(self, cam_id=None, *args):
        self.single_flag = True
        
        if not cam_id and not self.prev_cam_id:
            cam_id = self.default_camera_index
        if self.prev_cam_id:
            cam_id = self.prev_cam_id
        
        self.image = self.view.ids.camera_screen.ids.camera_canvas
        if platform == 'win32':
            self.capture = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        elif platform in ['linux', 'linux2']:
            self.capture = cv2.VideoCapture(cam_id, cv2.CAP_V4L)
        self.capture.set(3, self.FRAME_WIDTH)
        self.capture.set(4, self.FRAME_HEIGHT)
        self.video_event = Clock.schedule_interval(partial(self.load_video, "single"), 1.0 / 60.0)
    
    def toggle_camera(self):
        '''
        Toggles stereo cameras on and off
        '''
        if self.video_event:
            self.stop_camera()
            self.video_event = None
            self.view.ids.camera_screen.ids.camera_menu.disabled = True
            self.view.ids.camera_screen.ids.camera_capture.disabled = True
            self.view.ids.camera_screen.ids.camera_toggle.icon = "camera"
        else :
            self.start_camera()
            self.view.ids.camera_screen.ids.camera_menu.disabled = False
            self.view.ids.camera_screen.ids.camera_capture.disabled = False
            self.view.ids.camera_screen.ids.camera_toggle.icon = "camera-off"

    def start_stereo_cameras(self):
        self.stereo_flag = True
        
        if len(self.cameras) >= 2:
            self.left_camera = self.view.ids.stereo_camera_screen.ids.left_camera
            if platform == 'win32':
                self.left_capture = cv2.VideoCapture(self.left_cam_index, cv2.CAP_DSHOW)
            elif platform in ['linux', 'linux2']:
                self.left_capture = cv2.VideoCapture(self.left_cam_index, cv2.CAP_V4L)
            self.left_capture.set(3, self.FRAME_WIDTH)
            self.left_capture.set(4, self.FRAME_HEIGHT)
            self.left_video_event = Clock.schedule_interval(partial(self.load_video, "left"), 1.0 / 66.0)

            self.right_camera = self.view.ids.stereo_camera_screen.ids.right_camera
            if platform == 'win32':
                self.right_capture = cv2.VideoCapture(self.right_cam_index, cv2.CAP_DSHOW)
            elif platform in ['linux', 'linux2']:
                self.right_capture = cv2.VideoCapture(self.right_cam_index, cv2.CAP_V4L)
            self.right_capture.set(3, self.FRAME_WIDTH)
            self.right_capture.set(4, self.FRAME_HEIGHT)
            self.right_video_event = Clock.schedule_interval(partial(self.load_video, "right"), 1.0 / 66.0)
        else:
            toast("You have less than 2 cameras connected!")


    def stop_stereo_cameras(self, explore=False, cam=None):
        self.stereo_flag = False

        self.left_video_event.cancel()
        self.right_video_event.cancel()
        self.right_capture.release()
        self.left_capture.release()
        Clock.unschedule(partial(self.load_video, "left"))
        Clock.unschedule(partial(self.load_video, "right"))

        texture = Texture.create(size=(self.FRAME_WIDTH, self.FRAME_HEIGHT))
        texture.blit_buffer(bytes([255, 255, 255] * self.FRAME_WIDTH * self.FRAME_HEIGHT), colorfmt='rgb', bufferfmt='ubyte')

        self.left_camera.texture = texture
        self.right_camera.texture = texture

    
    def toggle_stereo_cameras(self):
        '''
        Toggles stereo cameras on and off
        '''
        if self.left_video_event and self.right_video_event:
            self.stop_stereo_cameras()
            self.left_video_event = None
            self.right_video_event = None
            self.view.ids.stereo_camera_screen.ids.stereo_camera_swap.disabled = True
            self.view.ids.stereo_camera_screen.ids.stereo_camera_capture.disabled = True
            self.view.ids.stereo_camera_screen.ids.camera_state_toggle.icon = "camera"
        else :
            self.start_stereo_cameras()
            self.view.ids.stereo_camera_screen.ids.stereo_camera_swap.disabled = False
            self.view.ids.stereo_camera_screen.ids.stereo_camera_capture.disabled = False
            self.view.ids.stereo_camera_screen.ids.camera_state_toggle.icon = "camera-off"
    

    def get_view(self) -> View.MainScreen.main_screen:
        return self.view