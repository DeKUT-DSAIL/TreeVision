import os
import importlib

from kivy.core.window import Window

from kivymd.uix.filemanager import MDFileManager
from kivymd.toast import toast

import View.ExtractScreen.extract_screen

# We have to manually reload the view module in order to apply the
# changes made to the code on a subsequent hot reload.
# If you no longer need a hot reload, you can delete this instruction.
importlib.reload(View.ExtractScreen.extract_screen)
from kivy.uix.image import Image



class ExtractScreenController:
    """
    The `ExtractScreenController` class represents a controller implementation.
    Coordinates work of the view with the model.
    The controller implements the strategy pattern. The controller connects to
    the view to control its actions.
    """
    image_path = None

    def __init__(self):
        self.view = View.ExtractScreen.extract_screen.ExtractScreenView(controller=self)
        # Window.bind(on_keyboard = self.events)
        self.manager_open = False
        self.file_manager = None
        self.folder_paths = {}

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
        self.file_manager.show(os.path.expanduser("~"))
        self.manager_open = True
    
    def store_folder_paths(self, path, button_id):
        self.folder_paths[button_id] = path
        self.exit_manager()
        toast(path)
        print(self.folder_paths)
    
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
    
    def show_next_image(self):
        pass
