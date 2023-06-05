import os
import time
import importlib
from glob import glob

from kivy.core.window import Window
from kivy.metrics import dp
from kivy.properties import StringProperty

from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.list import OneLineIconListItem
from kivymd.toast import toast
from kivy.clock import Clock

import View.ExtractScreen.extract_screen
from . import algorithms
import cv2
import numpy as np

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
    num_of_images = 0

    ASSET_DIR = 'assets'
    PROJECT_DIR = 'assets/projects'
    DISPARITY_MAPS_DIR = ''

    def __init__(self):
        self.view = View.ExtractScreen.extract_screen.ExtractScreenView(controller=self)
        # Window.bind(on_keyboard = self.events)
        self.manager_open = False
        self.file_manager = None
        self.folder_paths = {"left": "", "right": ""}

        self.parameter_menu_items = [
            {
                "viewclass": "OneLineListItem",
                "text": "DBH",
                "height": dp(56),
                "on_release": lambda x="DBH": self.set_item(self.parameter_menu, self.view.parameter_dropdown_item, x),
            },
            {
                "viewclass": "OneLineListItem",
                "text": "CD",
                "height": dp(56),
                "on_release": lambda x="CD": self.set_item(self.parameter_menu, self.view.parameter_dropdown_item, x),
            },
            {
                "viewclass": "OneLineListItem",
                "text": "TH",
                "height": dp(56),
                "on_release": lambda x="TH": self.set_item(self.parameter_menu, self.view.parameter_dropdown_item, x),
            }
        ]

        self.segmentation_menu_items = [
            {
                "viewclass": "OneLineListItem",
                "text": "Raw CNN",
                "height": dp(56),
                "on_release": lambda x="Raw CNN": self.set_item(self.segmentation_menu, self.view.segmentation_dropdown_item, x),
            },
            {
                "viewclass": "OneLineListItem",
                "text": "U-Net",
                "height": dp(56),
                "on_release": lambda x="U-Net": self.set_item(self.segmentation_menu, self.view.segmentation_dropdown_item, x),
            }
        ]

        self.parameter_menu = MDDropdownMenu(
            caller=self.view.parameter_dropdown_item,
            items=self.parameter_menu_items,
            position="center",
            background_color='brown',
            width_mult=2,
        )
        self.parameter_menu.bind()

        self.segmentation_menu = MDDropdownMenu(
            caller=self.view.segmentation_dropdown_item,
            items=self.segmentation_menu_items,
            position="center",
            background_color='brown',
            width_mult=2,
        )
        self.segmentation_menu.bind()
    
    def set_item(self, menu, dropdown_item, text_item):
        dropdown_item.set_item(text_item)
        self.view.parameter_dropdown_item.text = text_item
        menu.dismiss()

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
        '''
        Stores the paths the folders containing the left and right images in a dictionary with keys "left" and "right" \n
        The folders are selected using file manager buttons on the user interface. There are separate buttons for selecting \n
        the left and right folders and they have the IDs "left" and "right" respectively.

        @param path The path to the folder
        @param button_id The ID of the folder selection button. It takes one of the values "left" or "right"
        '''

        self.folder_paths[button_id] = path
        self.exit_manager()
        toast(path)
    
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
        '''
        Returns two lists for all paths to the images contained in the left and right folders. The left and right folder paths are taken from the dictionary with the keys "left" and "right"
        '''

        left_ims = glob(self.folder_paths['left'] + "/*.jpg")
        right_ims = glob(self.folder_paths['right'] + "/*.jpg")

        self.num_of_images = len(left_ims)

        verify_required = self.view.verify_checkbox.active

        if verify_required and self.verify_images(left_ims, right_ims):
            return (left_ims, right_ims)
        elif verify_required and not self.verify_images(left_ims, right_ims):
            toast("Number of Left and Right Images Not equal")
        elif not verify_required and (len(left_ims) > 0 and len(right_ims) > 0):
            toast("Number of Left and Right images not verified!")
            return (left_ims, right_ims)
        

    def on_button_press(self, instance):
        '''
        Enables scrolling forward and backward to facilitate viewing the corresponding images in the left and right folders. There are two buttons on the user interface, one for scrolling forward and another for scrolling backward.\n
        @param instance The instance of the button pressed to scroll. It takes the values "next" or "previous"
        '''

        if self.num_of_images == 0:
            toast("Select Left and Right image folders first")
            return
        if instance == 'next':
            self.image_index = (self.image_index + 1) % self.num_of_images
            return True
        elif instance == 'previous':
            self.image_index = (self.image_index - 1) % self.num_of_images
            return True
    

    def show_next_image(self, button_id):
        '''
        Displays the next image in the sequence once the scroll buttons are clicked
        @param button_id The ID of the scroll button clicked. It takes the values "next" or "previous"
        '''

        left, right, _ = self.load_stereo_images()

        if self.on_button_press(button_id):
            self.view.left_im.source = left[self.image_index]
            self.view.right_im.source = right[self.image_index]


    def create_disparity_directory(self):
        '''
        Creates a directory in the "assets" folder of the app for the project. This is the directory where the extracted disparity maps will be saved
        '''

        project = self.view.project_name.text
        if project == '':
            toast("Please provide the project name!")
        else:
            path = os.path.join(self.PROJECT_DIR, f'{project}/disparity_maps')
            if not os.path.exists(path):
                os.makedirs(path)
                self.DISPARITY_MAPS_DIR = path
            else:
                pass


    def save_and_display_disparity(self, left_img_path=None, right_img_path=None):
        '''
        Saves the extracted disparity map in the project folder and displays it in the user interface on the position initially occupied by the right image.
        
        @param left_img_path The path to the left image
        @param right_img_path The path to the right image
        '''

        left_img_path = self.view.left_im.source
        right_img_path = self.view.right_im.source

        main_folder_path = os.path.dirname(os.path.dirname(left_img_path))
        left_img_filename = os.path.basename(left_img_path)
        mask_path =  main_folder_path + '/masks/' + left_img_filename.split(".")[0] + "_mask.png"

        left = cv2.imread(left_img_path, 0)
        right = cv2.imread(right_img_path, 0)
        mask = cv2.imread(mask_path, 0)
        kernel = np.ones((3,3),np.uint8)

        dmap = extract(left, right, mask, kernel)
        dmap_filename = left_img_path.split('/')[-1].split('.')[0] + '_disparity.jpg'
        dmap_path = os.path.join(self.DISPARITY_MAPS_DIR, dmap_filename)

        cv2.imwrite(dmap_path, dmap)
        self.view.right_im.source = dmap_path


    def on_extract(self):
        '''
        Called when the "Extract" button is pressed on the user interface
        '''

        self.create_disparity_directory()
        self.save_and_display_disparity()
        self.compute_parameter()


    def on_batch_extract(self, dt):
        
        self.create_disparity_directory()

        left_ims, right_ims = self.load_stereo_images()
        left_img = left_ims[self.image_index]
        right_img = right_ims[self.image_index]

        self.view.left_im.source = left_img
        self.view.right_im.source = right_img

        self.save_and_display_disparity(
                left_img_path=left_img,
                right_img_path=right_img
            )

        if self.image_index < len(left_ims) - 1:
            self.image_index += 1
        elif self.image_index == len(left_ims) - 1:
            toast("Batch extraction complete")
    
    def update_on_batch_extract(self):
        Clock.schedule_interval(self.on_batch_extract, 2)

    
    def compute_parameter(self):
        parameter = self.view.parameter_dropdown_item.text
        print(f"Parameter: {parameter}")
        dmap = cv2.imread(self.view.right_im.source, 0)
        
        if parameter == "DBH":
            algorithms.compute_dbh(dmap)
        elif parameter == "CD":
            algorithms.compute_cd(dmap)
        elif parameter == "TH":
            algorithms.compute_th(dmap)



def extract(left_im, right_im, mask, sel):
    '''
    Extracts the disparity map and returns it
    '''
    dmap = algorithms.compute_depth_map(
        imgL = left_im,
        imgR = right_im,
        mask = mask,
        sel= sel
    )

    return dmap['R']

class IconListItem(OneLineIconListItem):
    icon = StringProperty()