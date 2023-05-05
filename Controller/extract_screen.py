import os
import importlib
from glob import glob

from kivy.core.window import Window
from kivy.metrics import dp
from kivy.properties import StringProperty

from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.list import OneLineIconListItem
from kivymd.toast import toast

import View.ExtractScreen.extract_screen

# We have to manually reload the view module in order to apply the
# changes made to the code on a subsequent hot reload.
# If you no longer need a hot reload, you can delete this instruction.
importlib.reload(View.ExtractScreen.extract_screen)
from kivy.uix.image import Image

from kivymd.uix.dropdownitem.dropdownitem import MDDropDownItem

class ExtractScreenController:
    """
    The `ExtractScreenController` class represents a controller implementation.
    Coordinates work of the view with the model.
    The controller implements the strategy pattern. The controller connects to
    the view to control its actions.
    """

    image_index = 0
    image_path = None

    def __init__(self):
        self.view = View.ExtractScreen.extract_screen.ExtractScreenView(controller=self)
        # Window.bind(on_keyboard = self.events)
        self.manager_open = False
        self.file_manager = None
        self.folder_paths = {"left": "", "right": ""}

        self.parameter_menu_items = [
            {
                "viewclass": "IconListItem",
                "icon": "git",
                "text": "DBH",
                "height": dp(56),
                "on_release": lambda x="DBH": self.set_item(x),
            },
            {
                "viewclass": "IconListItem",
                "icon": "git",
                "text": "CD",
                "height": dp(56),
                "on_release": lambda x="CD": self.set_item(x),
            },
            {
                "viewclass": "IconListItem",
                "icon": "git",
                "text": "TH",
                "height": dp(56),
                "on_release": lambda x="TH": self.set_item(x),
            }
        ]

        self.menu = MDDropdownMenu(
            caller=self.view.dropdown_item,
            items=self.parameter_menu_items,
            position="center",
            width_mult=2,
        )
        self.menu.bind()
    
    def set_item(self, text_item):
        self.view.dropdown_item.set_item(text_item)
        self.menu.dismiss()

    def get_view(self) -> View.ExtractScreen.extract_screen:
        return self.view
    
    def toggle_cameras(self, widget):
        if widget.state == 'normal':
            self.view.camera.play = False
        else:
            self.view.camera.play = True
    

    def file_manager_open(self, button_id):
        self.file_manager = MDFileManager(
            selector = "folder",
            exit_manager = self.exit_manager,
            select_path = lambda path: self.store_folder_paths(path, button_id)
        )
        self.file_manager.show(os.path.expanduser("/"))
        self.manager_open = True
    
    def store_folder_paths(self, path, button_id):
        self.folder_paths[button_id] = path
        self.exit_manager()
        toast(path)
        self.load_stereo_images()
    
    def exit_manager(self, *args):
        '''Called when the user reaches the root of the directory tree.'''

        self.manager_open = False
        self.file_manager.close()
    
    def events(self, instance, keyboard, keycode, text, modifiers):
        '''Called when buttons are pressed on the mobile device.'''

        if keyboard in (1001, 27):
            if self.manager_open:
                self.file_manager.back()
        return True

    def verify_images(self, left_ims, right_ims):
        return len(left_ims) == len(right_ims)

    def load_stereo_images(self):
        left_ims = glob(self.folder_paths['left'] + "/*.jpg")
        right_ims = glob(self.folder_paths['right'] + "/*.jpg")

        verify_required = self.view.verify_checkbox.active

        if verify_required and self.verify_images(left_ims, right_ims):
            return (left_ims, right_ims)
        elif verify_required and not self.verify_images(left_ims, right_ims):
            toast("Number of Left and Right Images Not equal")
        elif not verify_required and (len(left_ims) > 0 and len(right_ims) > 0):
            toast("Number of Left and Right images not verified!")
            return (left_ims, right_ims)
        

    def on_button_press(self, instance):
        if instance == 'next':
            self.image_index += 1
            return True
        elif instance == 'previous' and self.image_index > 0:
            self.image_index -= 1
            return True
        elif instance == 'previous' and self.image_index == 0:
            toast("This is the first image")
            return False


    def show_next_image(self, button_id):
        left, right = self.load_stereo_images()

        if self.on_button_press(button_id):
            if len(left) == 0:
                toast("Select the left and right images folder first")
            elif len(right) == 0:
                toast("Select the left and right images folder first")
            else:
                self.view.left_im.source = left[self.image_index]
                self.view.right_im.source = right[self.image_index]


class IconListItem(OneLineIconListItem):
    icon = StringProperty()