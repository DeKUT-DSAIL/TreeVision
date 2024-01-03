import importlib
import os

from kivymd.toast import toast
from kivymd.uix.filemanager import MDFileManager

import View.UploadScreen.upload_screen

# We have to manually reload the view module in order to apply the
# changes made to the code on a subsequent hot reload.
# If you no longer need a hot reload, you can delete this instruction.
importlib.reload(View.UploadScreen.upload_screen)


class UploadScreenController:
    """
    The `MainScreenController` class represents a controller implementation.
    Coordinates work of the view with the model.
    The controller implements the strategy pattern. The controller connects to
    the view to control its actions.
    """

    def __init__(self):
        self.view = View.UploadScreen.upload_screen.UploadScreenView(controller=self)

        self.manager_open = False
        self.file_manager = MDFileManager(
            exit_manager=self.exit_manager,
            select_path=self.select_path,
            ext=['.jpg', '.png', '.jpeg'],
            preview=True,
        )

    def switch_screen(self, screen_name):
        if isinstance(screen_name, str):
            self.view.manager_screens.current = screen_name
        else:
            self.view.manager_screens.switch_to(screen_name)

    def file_manager_open(self):
        self.file_manager.show(os.getcwd())
        self.manager_open = True

    def select_path(self, path):
        """ called when you click on file name """
        image_screen = self.view.manager_screens.get_screen("image-screen")
        image_screen.ids.image_section.image_source = path
        self.switch_screen("image-screen")
        self.exit_manager()
        image_path = path.split("/")[-1]
        toast(image_path)

    def exit_manager(self, *args):
        """ Called when user reaches roo of directory tree """
        self.manager_open = False
        self.file_manager.close()

    def get_view(self) -> View.UploadScreen.upload_screen:
        return self.view
