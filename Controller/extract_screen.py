import os
import time
import importlib
import csv
from glob import glob

from kivy.core.window import Window
from kivy.metrics import dp
from kivy.properties import StringProperty

from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.list import OneLineIconListItem
from kivymd.uix.button import MDIconButton
from kivymd.toast import toast
from kivy.clock import Clock

import View.ExtractScreen.extract_screen
from . import algorithms
import cv2
import numpy as np
import pandas as pd

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
    PROJECT_DIR = os.path.join(ASSET_DIR, 'projects')
    DISPARITY_MAPS_DIR = None
    RESULTS_DIR = None
    IMAGES_DIR = None
    FILE_MANAGER_SELECTOR = 'folder'

    def __init__(self):
        self.view = View.ExtractScreen.extract_screen.ExtractScreenView(controller=self)
        # Window.bind(on_keyboard = self.events)
        self.manager_open = False
        self.file_manager = None

        self.file_manager = MDFileManager(
            selector = self.FILE_MANAGER_SELECTOR,
            exit_manager = self.exit_manager,
            select_path = self.select_path
        )

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
        self.toggle_scrolling_icons()

    

    def toggle_scrolling_icons(self):

        if self.IMAGES_DIR == None:
            self.view.previous_arrow.opacity = 0
            self.view.next_arrow.opacity = 0
        else:
            self.view.previous_arrow.opacity = 1
            self.view.next_arrow.opacity = 1
            self.view.previous_arrow.on_release = lambda: self.show_next_image('previous')
            self.view.next_arrow.on_release = lambda: self.show_next_image('next')


    def set_item(self, menu, dropdown_item, text_item):
        dropdown_item.set_item(text_item)
        self.view.parameter_dropdown_item.text = text_item
        menu.dismiss()



    def get_view(self) -> View.ExtractScreen.extract_screen:
        return self.view
    
    

    def file_manager_open(self, selector):
        '''
        Opens the file manager when the triggering event in the user interface happens
        '''
        self.FILE_MANAGER_SELECTOR = selector
        
        self.file_manager = MDFileManager(
            selector = self.FILE_MANAGER_SELECTOR,
            exit_manager = self.exit_manager,
            select_path = self.select_path
        )

        self.file_manager.show(os.path.expanduser("."))
        self.manager_open = True

    

    def select_path(self, path: str):
        '''
        It will be called when you click on the file name
        or the catalog selection button.

        @param path: path to the selected directory or file;
        '''

        self.IMAGES_DIR = path
        self.toggle_scrolling_icons()
        self.exit_manager()
        toast(self.IMAGES_DIR)


    
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
        
        left_ims = glob(os.path.join(self.IMAGES_DIR, 'left/*.jpg'))
        right_ims = glob(os.path.join(self.IMAGES_DIR, 'right/*.jpg'))

        self.num_of_images = len(left_ims)

        if self.verify_images(left_ims, right_ims):
            return (left_ims, right_ims)
        
        toast("Number of Left and Right Images Not equal")
    


    def on_button_press(self, instance):
        '''
        Enables scrolling forward and backward to facilitate viewing the corresponding images in the left and right folders. There are two buttons on the user interface, one for scrolling forward and another for scrolling backward.\n
        
        @param instance: The instance of the button pressed to scroll. It takes the values "next" or "previous"
        '''

        if instance == 'next':
            self.image_index = (self.image_index + 1) % self.num_of_images
            return True
        elif instance == 'previous':
            self.image_index = (self.image_index - 1) % self.num_of_images
            return True
    


    def show_next_image(self, button_id):
        '''
        Displays the next image in the sequence once the scroll buttons are clicked

        @param button_id: The ID of the scroll button clicked. It takes the values "next" or "previous"
        '''


        left, right = self.load_stereo_images()
        left = sorted(left)
        right = sorted(right)

        if self.on_button_press(button_id):
            self.view.left_im.source = left[self.image_index]
            self.view.right_im.source = right[self.image_index]



    def create_project_directories(self):
        '''
        Creates a directory in the "assets" folder of the app for the project. This is the directory where the extracted disparity maps will be saved
        '''

        project = self.view.project_name.text

        if project == '':
            toast("Please provide a project name!")
            return False
        
        else:
            dmaps_path = os.path.join(self.PROJECT_DIR, f'{project}/disparity_maps')
            results_path = os.path.join(self.PROJECT_DIR, f'{project}/results')

            self.DISPARITY_MAPS_DIR = dmaps_path
            self.RESULTS_DIR = results_path
            
            os.makedirs(dmaps_path) if not os.path.exists(results_path) else None
            os.makedirs(results_path) if not os.path.exists(results_path) else None

            results_file = os.path.join(self.RESULTS_DIR, 'results.csv')

            if not os.path.exists(results_file):
                results_df = pd.DataFrame(columns=['DBH', 'CD', 'TH'])
                results_df.index.name = 'Filename'
                results_df.to_csv(results_file)
            
            return True



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
        masks_folder_path = os.path.join(main_folder_path, 'masks')
        mask_filename = left_img_filename.split(".")[0] + "_mask.png"

        mask_path =  os.path.join(masks_folder_path, mask_filename)

        left = cv2.imread(left_img_path, 0)
        right = cv2.imread(right_img_path, 0)
        mask = cv2.imread(mask_path, 0)
        kernel = np.ones((3,3), np.uint8)

        dmap = algorithms.extract(left, right, mask, kernel)
        dmap_filename = left_img_path.split('/')[-1].split('.')[0] + '_disparity.jpg'
        dmap_path = os.path.join(self.DISPARITY_MAPS_DIR, dmap_filename)

        cv2.imwrite(dmap_path, dmap)
        self.view.right_im.source = dmap_path
        

    def on_extract(self):
        '''
        Called when the "Extract" button is pressed on the user interface
        '''

        if self.create_project_directories():
            self.save_and_display_disparity()
            parameter, value = self.compute_parameter()

            left_filename = os.path.basename(self.view.left_im.source)
            new_row = {parameter: round(value, 2)}

            results_file = os.path.join(self.RESULTS_DIR, 'results.csv')

            results_df = pd.read_csv(results_file, index_col='Filename')
            results_df.loc[left_filename] = new_row
            print(results_df)
            results_df.to_csv(results_file)
        
        else:
            toast("Provide a project name to extract measurements!")



    def on_batch_extract(self, dt):
        
        self.create_project_directories()

        left_ims, right_ims = self.load_stereo_images()
        left_ims = sorted(left_ims)
        right_ims = sorted(right_ims)

        left_img = left_ims[self.image_index]
        right_img = right_ims[self.image_index]

        self.view.left_im.source = left_img
        self.view.right_im.source = right_img

        self.save_and_display_disparity(
                left_img_path=left_img,
                right_img_path=right_img
            )

        parameter, value = self.compute_parameter()

        left_filename = os.path.basename(self.view.left_im.source)

        new_row = {parameter: round(value, 2)}
        results_file = os.path.join(self.RESULTS_DIR, 'results.csv')

        results_df = pd.read_csv(results_file, index_col='Filename')
        results_df.loc[left_filename] = new_row
        results_df.to_csv(results_file)

        if self.image_index < len(left_ims) - 1:
            self.image_index += 1
        else:
            toast('Batch extraction complete')
            self.unschedule_batch_extraction()
    


    def update_on_batch_extract(self):
        Clock.schedule_interval(self.on_batch_extract, 2)
    


    def unschedule_batch_extraction(self):
        Clock.unschedule(self.on_batch_extract)

    

    def compute_parameter(self):
        parameter = self.view.parameter_dropdown_item.text
        dmap = cv2.imread(self.view.right_im.source, 0)
        
        if parameter == "DBH":
            return parameter, algorithms.compute_dbh(dmap)
        elif parameter == "CD":
            return parameter, algorithms.compute_cd(dmap)
        elif parameter == "TH":
            return parameter, algorithms.compute_th(dmap)



class IconListItem(OneLineIconListItem):
    icon = StringProperty()