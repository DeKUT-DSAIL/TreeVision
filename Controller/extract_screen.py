import os
import time
import importlib
import csv
import matplotlib.pyplot as plt
from glob import glob
from sys import platform

from kivy.core.window import Window
from kivy.metrics import dp
from kivy.utils import rgba
from kivy.properties import StringProperty

from kivymd.app import MDApp
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.list import OneLineIconListItem
from kivymd.uix.button import MDFlatButton, MDRectangleFlatButton, MDRaisedButton
from kivymd.uix.label import MDLabel
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.dialog import MDDialog
from kivymd.toast import toast
from kivy.clock import Clock

import View.ExtractScreen.extract_screen
from . import algorithms
import cv2
import numpy as np
import pandas as pd
import webbrowser
from sklearn.metrics import mean_squared_error

from View.ExtractScreen.extract_screen import RefreshConfirm
from View.ExtractScreen.extract_screen import InfoPopupModal

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
    
    app = MDApp.get_running_app()
    dialog = None
    info_popup_modal = None

    image_index = 0
    num_of_images = 0

    ASSET_DIR = 'assets'
    PROJECT_DIR = os.path.join(ASSET_DIR, 'projects')
    DISPARITY_MAPS_DIR = None
    ANNOTATED_IMAGES_DIR = None
    RESULTS_DIR = None
    IMAGES_DIR = 'test/full_trees'
    FILE_MANAGER_SELECTOR = 'folder'
    SELECT_BUTTON_ID = None
    THIS_PROJECT = None

    CONFIG_FILE_PATH = 'configs/stereo_kieni.yml'
    REF_PARAMS_FILE = None
    DIAG_FIELD_OF_VIEW = None
    HORZ_FIELD_OF_VIEW = None
    VERT_FIELD_OF_VIEW = None

    LEFT_IMS = None
    RIGHT_IMS = None

    def __init__(self):
        self.view = View.ExtractScreen.extract_screen.ExtractScreenView(controller=self)
        Window.bind(on_keyboard = self.events)
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
                "on_release": lambda x="DBH": self.set_item(self.parameter_menu, self.view.parameter_dropdown_item, x),
            },
            {
                "viewclass": "OneLineListItem",
                "text": "CD & TH",
                "on_release": lambda x="CD & TH": self.set_item(self.parameter_menu, self.view.parameter_dropdown_item, x),
            }
        ]

        self.segmentation_menu_items = [
            {
                "viewclass": "OneLineListItem",
                "text": "Masks",
                "on_release": lambda x="Masks": self.set_item(self.segmentation_menu, self.view.segmentation_dropdown_item, x),
            }
        ]

        self.rectification_menu_items = [
            {
                "viewclass": "OneLineListItem",
                "text": "Yes",
                "on_release": lambda x="Yes": self.set_item(self.rectification_menu, self.view.ids.rectification_dropdown_item, x),
            },
            {
                "viewclass": "OneLineListItem",
                "text": "No",
                "on_release": lambda x="No": self.set_item(self.rectification_menu, self.view.ids.rectification_dropdown_item, x),
            }
        ]

        self.format_menu_items = [
            {
                "viewclass": "OneLineListItem",
                "text": "JPG",
                "on_release": lambda x="Yes": self.set_item(self.format_menu, self.view.ids.rectification_dropdown_item, x),
            },
            {
                "viewclass": "OneLineListItem",
                "text": "No",
                "on_release": lambda x="No": self.set_item(self.format_menu, self.view.ids.rectification_dropdown_item, x),
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

        self.rectification_menu = MDDropdownMenu(
            caller=self.view.ids.segmentation_dropdown_item,
            items=self.rectification_menu_items,
            position="center",
            background_color='brown',
            width_mult=2,
        )
        self.rectification_menu.bind()

        self.set_display_images()
        self.toggle_scrolling_icons()
        self.initialize_sgbm_values()
    


    def set_display_images(self):
        '''
        Sets the display images
        '''
        if platform == "win32":
            self.view.ids.left_im.source = "assets\\images\\extraction\\FT01_IMG_20230309_103936_LEFT.jpg"
            self.view.ids.right_im.source = "assets\\images\\extraction\\FT01_IMG_20230309_103936_RIGHT.jpg"
        elif platform in ["linux", "linux2"]:
            self.view.ids.left_im.source = "assets/images/extraction/FT01_IMG_20230309_103936_LEFT.jpg"
            self.view.ids.right_im.source = "assets/images/extraction/FT01_IMG_20230309_103936_RIGHT.jpg"

    

    def toggle_scrolling_icons(self):
        '''
        Toggles the buttons for scrolling the images left and right based on whether the project path has been selected
        or not. The buttons are toggled off if a project path with multiple images has not been selected.
        '''

        if self.num_of_images == 0:
            self.view.previous_arrow.opacity = 0
            self.view.next_arrow.opacity = 0
        else:
            self.view.previous_arrow.opacity = 1
            self.view.next_arrow.opacity = 1
            self.view.previous_arrow.on_release = lambda: self.show_next_image('previous')
            self.view.next_arrow.on_release = lambda: self.show_next_image('next')


    def set_item(self, menu, dropdown_item, text_item):
        dropdown_item.set_item(text_item)
        dropdown_item.text = text_item
        menu.dismiss()



    def get_view(self) -> View.ExtractScreen.extract_screen:
        return self.view
    
    

    def file_manager_open(self, selector, button_id):
        '''
        Opens the file manager when the triggering event in the user interface happens
        '''
        self.FILE_MANAGER_SELECTOR = selector
        self.SELECT_BUTTON_ID = button_id
        
        self.file_manager = MDFileManager(
            selector = self.FILE_MANAGER_SELECTOR,
            exit_manager = self.exit_manager,
            select_path = self.select_path
        )

        self.file_manager.show(os.path.expanduser("~"))
        self.manager_open = True

    

    def select_path(self, path: str):
        '''
        It will be called when you click on the file name
        or the catalog selection button.

        @param path: path to the selected directory or file;
        '''

        if self.FILE_MANAGER_SELECTOR == 'folder':
            self.IMAGES_DIR = path 
            self.load_stereo_images()
            self.create_log_widget(f"Project images directory has been selected.\nIMAGES DIRECTORY PATH: {path}")
        elif self.FILE_MANAGER_SELECTOR == 'file':
            if self.SELECT_BUTTON_ID == 1:
                self.CONFIG_FILE_PATH = path
                self.create_log_widget(f"Camera configuration file has been selected.\nCAMERA CONFIGURATION FILE PATH: {path}")
            elif self.SELECT_BUTTON_ID == 2:
                self.REF_PARAMS_FILE = path
                self.create_log_widget(f"Reference parameters file has been selected.\nREFERENCE PARAMETERS FILE PATH: {path}")
        
        self.toggle_scrolling_icons()
        self.exit_manager()


    
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
    


    def initialize_sgbm_values(self):
        '''
        Initializes the parameters of the SGBM algorithm
        '''
        self.view.ids.min_disp.text = "0"
        self.view.ids.num_disp.text = "128"
        self.view.ids.block_size.text = "11"
        self.view.ids.uniqueness_ratio.text = "10"
        self.view.ids.speckle_window_size.text = "100"
        self.view.ids.speckle_range.text = "2"
        self.view.ids.disp_max_diff.text = "5"
    


    def verify_images(self, left_ims, right_ims):
        return len(left_ims) == len(right_ims)
    


    def load_stereo_images(self):
        '''
        Returns two lists for all paths to the left and right images. The left and right folder paths are taken from the dictionary with the keys "left" and "right"
        '''
        left_patterns = ['*LEFT.jpg', '*left.jpg', '*LEFT.png', '*left.png']
        right_patterns = ['*RIGHT.jpg', '*right.jpg', '*RIGHT.png', '*right.png']

        left_ims = []
        right_ims = []

        for pattern in left_patterns:
            left_ims.extend(glob(os.path.join(self.IMAGES_DIR, pattern)))
        
        for pattern in right_patterns:
            right_ims.extend(glob(os.path.join(self.IMAGES_DIR, pattern)))

        left_ims = sorted(list(set(left_ims)))
        right_ims = sorted(list(set(right_ims)))

        self.LEFT_IMS = left_ims
        self.RIGHT_IMS = right_ims

        self.num_of_images = len(left_ims)
        self.view.ids.progress_bar.max = self.num_of_images

        if self.verify_images(left_ims, right_ims):
            self.view.ids.left_im.source = left_ims[0]
            self.view.ids.right_im.source = right_ims[0]
            self.view.ids.batch_extract_btn.disabled = False
        
        else:
            self.create_log_widget(text = "Number of Left and Right Images Not equal!", color=(1,0,0,1))
    


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

        if self.on_button_press(button_id):
            self.view.left_im.source = self.LEFT_IMS[self.image_index]
            self.view.right_im.source = self.RIGHT_IMS[self.image_index]



    def create_project_directories(self):
        '''
        Creates a directory in the "assets" folder of the app for the project. This is the directory where the extracted disparity maps as well as a CSV file containing the extracted parameters will be saved
        '''

        project = self.view.project_name.text 
        self.THIS_PROJECT = project
        project_path = os.path.join(self.PROJECT_DIR, f'{project}')
        dmaps_path = os.path.join(project_path, 'disparity_maps')
        results_path = os.path.join(project_path, 'results')
        annotated_images_path = os.path.join(project_path, 'annotated')

        self.DISPARITY_MAPS_DIR = dmaps_path
        self.RESULTS_DIR = results_path
        self.ANNOTATED_IMAGES_DIR = annotated_images_path

        if not os.path.exists(project_path):
            for path in (dmaps_path, results_path, annotated_images_path):
                os.makedirs(path, exist_ok=True)
            self.create_log_widget(text="Project folders have been created!")

        parameter = self.view.parameter_dropdown_item.text
        parameters_dict = {
            'DBH': (f'results_{self.THIS_PROJECT}_dbh.csv', ['Ref_DBH', 'Ex_DBH', 'AE_DBH (cm)', 'APE_DBH (%)']),
            'CD & TH': (f'results_{self.THIS_PROJECT}_cd_th.csv', ['Ref_TH', 'Ex_TH', 'AE_TH (cm)', 'APE_TH (%)', 'Ref_CD', 'Ex_CD', 'AE_CD (cm)', 'APE_CD (%)'])
        }

        if parameter in parameters_dict:
            results_file, columns = parameters_dict[parameter]
            results_file = os.path.join(self.RESULTS_DIR, results_file)

            if not os.path.exists(results_file):
                results_df = pd.DataFrame(columns=columns)
                results_df.index.name = 'Filename'
                results_df.to_csv(results_file)
        else:
            self.create_log_widget(
                text="Please choose a parameter to extract.",
                color=(1, 0, 0, 1)
            )



    def compute_and_save_disparity(self):
        '''
        Saves the extracted disparity map in the project folder and displays it in the user interface on the position initially occupied by the right image.
        It returns the paths to the segmentation mask and the segmented disparity map
        '''

        left_img_path = self.view.left_im.source
        right_img_path = self.view.right_im.source
        rectified = self.view.ids.rectification_dropdown_item.text
        
        if rectified == "Yes":
            rec_status = True
        elif rectified == "No":
            rec_status = False

        folder_path = os.path.dirname(left_img_path)
        left_img_filename = os.path.basename(left_img_path)
        mask_filename = left_img_filename.split(".")[0] + "_mask.*"

        mask_path =  glob(os.path.join(folder_path, mask_filename))[0]

        left = cv2.imread(left_img_path, 0)
        right = cv2.imread(right_img_path, 0)
        mask = cv2.imread(mask_path, 0)
        kernel = np.ones((3,3), np.uint8)

        if self.verify_config_file():
            dmap = algorithms.extract(
                left_im = left, 
                right_im = right, 
                mask = mask,
                rectified = rec_status,
                sel = kernel, 
                config_file_path = self.CONFIG_FILE_PATH,
                min_disp = int(self.view.ids.min_disp.text),
                num_disp = int(self.view.ids.num_disp.text),
                block_size = int(self.view.ids.block_size.text),
                uniqueness_ratio = int(self.view.ids.uniqueness_ratio.text),
                speckle_window_size = int(self.view.ids.speckle_window_size.text),
                speckle_range = int(self.view.ids.speckle_range.text),
                disp_max_diff = int(self.view.ids.disp_max_diff.text)
            )
            
            # change to left_img_path.split('\\') for Windows
            if platform == "win32":
                dmap_filename = left_img_path.split('\\')[-1].split('.')[0] + '_disparity.jpg'
            elif platform == "linux" or platform == "linux2":
                dmap_filename = left_img_path.split('/')[-1].split('.')[0] + '_disparity.jpg'
            
            dmap_path = os.path.join(self.DISPARITY_MAPS_DIR, dmap_filename)

            cv2.imwrite(dmap_path, dmap)
            return dmap_path, mask_path
        else:
            return False
    


    def annotate_image(self, dmap_path, parameter, dfov, values_dict):
        '''
        Annotates an image by showing the location of its boundaries and superimposing the values of the estimated parameters
        '''
        K, _, _, _, _, _, _, _, T, _ = algorithms.load_camera_params(self.CONFIG_FILE_PATH)

        dmap = cv2.imread(dmap_path, 0)
        base_px = algorithms.pixel_of_interest(dmap, 'DBH')
        base_depth = algorithms.disp_to_dist(base_px)

        focal_length = K[0, 0]
        cx = K[0, 2]
        cy = K[1, 2]
        baseline = T[0, 0] / 1000

        base, top, left, right = algorithms.convex_hull(dmap)
        base_loc = (base[1] - 200, base[0])
        h, w = dmap.shape

        left_image = cv2.imread(self.view.left_im.source)
        B, G, R = cv2.split(left_image)

        B = algorithms.rectify(B, self.CONFIG_FILE_PATH, 'left')
        G = algorithms.rectify(G, self.CONFIG_FILE_PATH, 'left')
        R = algorithms.rectify(R, self.CONFIG_FILE_PATH, 'left')

        left_image = cv2.merge([B,G,R])

        if parameter.lower() == 'dbh':
            bh = algorithms.compute_bh(dmap, base_depth, baseline, focal_length, dfov, cx, cy)

            cols = np.nonzero(dmap[bh, :])[0]
            left_edge = (cols.min(), bh)
            right_edge = (cols.max(), bh)

            left_image = cv2.arrowedLine(left_image, (0, bh) ,left_edge, (0,0,255), 5)
            left_image = cv2.arrowedLine(left_image, (w-1, bh) ,right_edge, (0,0,255), 5)
            left_image = cv2.arrowedLine(left_image, (base[1]-200, bh), base_loc, (0,0,255), 5)

            left_image = cv2.putText(left_image, f'{values_dict["DBH"]}cm', (cols.min()+5, bh+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
            left_image = cv2.putText(left_image, '1.3m', (base[1]-180, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)

            left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)

            return left_image
        
        else:
            left_y, left_x = left
            right_y, right_x = right
            top_y, top_x = top
            base_y, base_x = base

            horz_arrow_y = int(np.mean([left_y, right_y]))
            text_center_x = int(np.mean([left_x, right_x]))
            text_center_y = int(np.mean([top_y, base_y]))

            left_image = cv2.arrowedLine(left_image, (left_x, horz_arrow_y) , (right_x, horz_arrow_y), (0,0,255), 5, tipLength=0.1)
            left_image = cv2.arrowedLine(left_image, (right_x, horz_arrow_y), (left_x, horz_arrow_y), (0,0,255), 5, tipLength=0.1)
            left_image = cv2.arrowedLine(left_image, (right_x + 20, base_y) , (right_x + 20, top_y), (0,0,255), 5)

            left_image = cv2.putText(left_image, f'{values_dict["CD"]}m', (text_center_x-75, horz_arrow_y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
            left_image = cv2.putText(left_image, f'{values_dict["TH"]}m', (right_x+50, text_center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

            left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)

            return left_image
    


    def verify_project_name(self):
        '''
        Verifies that the camera calibration file is available and contains all the necessary parameters
        '''
        if self.view.ids.project_name.text  == '':
            return False
        else:
            return True

    

    def verify_config_file(self):
        '''
        Verifies that the camera calibration file is available and contains all the necessary parameters
        '''
        if self.CONFIG_FILE_PATH == None:
            return False
        else:
            return True

        

    def on_extract(self):
        '''
        Called when the "Extract" button on the user interface is pressed
        '''
        if not self.verify_config_file():
            self.create_log_widget(text = "Missing camera configuration file path!", color=(1,0,0,1))
        
        elif not self.verify_project_name():
            self.create_log_widget(text = "Missing the project name! Please provide one to proceed.", color=(1,0,0,1))

        else:
            self.create_project_directories()
            self.DIAG_FIELD_OF_VIEW = np.float32(self.view.ids.dfov.text)
            dmap_path, mask_path = self.compute_and_save_disparity()
            
            parameter = self.view.parameter_dropdown_item.text

            parameters, values = self.compute_parameter(mask_path)
            values_dict = {}
            for i in range(len(parameters)):
                values_dict[parameters[i]] = values[i]

            annotated_image = self.annotate_image(dmap_path, parameter, self.DIAG_FIELD_OF_VIEW, values_dict)

            left_filename = os.path.basename(self.view.left_im.source)

            if platform == 'win32':
                annotated_image_name = left_filename.split('\\')[-1].split('.')[0] + '_annotated.jpg'
            elif platform in ['linux', 'linux2']:
                annotated_image_name = left_filename.split('/')[-1].split('.')[0] + '_annotated.jpg'
            
            annotated_image_path = os.path.join(self.ANNOTATED_IMAGES_DIR, annotated_image_name)
            cv2.imwrite(annotated_image_path, annotated_image)
            self.view.right_im.source = annotated_image_path

            self.display_parameters_on_logs(
                image = left_filename,
                parameters = parameters,
                values = values
            )

            new_row = {f"Ex_{k}": round(v*100, 2) for k,v in zip(parameters, values)}

            if parameter == 'DBH':
                results_file = os.path.join(self.RESULTS_DIR, f'results_{self.THIS_PROJECT}_dbh.csv')
            elif parameter == 'CD & TH':
                results_file = os.path.join(self.RESULTS_DIR, f'results_{self.THIS_PROJECT}_cd_th.csv')

            results_df = pd.read_csv(results_file, index_col='Filename')
            results_df.loc[left_filename] = new_row
            results_df.to_csv(results_file)



    def on_batch_extract(self, dt):
        '''
        This function performs batch extraction of tree parameters from all images in the selected images directory. 
        The parameters are saved in a CSV file in the 'results' subdirectory of the projects folder.
        Called when the "Batch extract" button on the user interface is pressed
        '''

        self.create_project_directories()
        
        left_ims, right_ims = self.LEFT_IMS, self.RIGHT_IMS

        left_img = left_ims[self.image_index]
        right_img = right_ims[self.image_index]

        self.view.left_im.source = left_img
        self.view.right_im.source = right_img

        self.DIAG_FIELD_OF_VIEW = np.float32(self.view.ids.dfov.text)

        dmap_path, mask_path = self.compute_and_save_disparity()
        
        self.view.right_im.source = dmap_path
        self.view.ids.progress_bar.value = self.image_index + 1

        parameters, values = self.compute_parameter(mask_path)

        left_filename = os.path.basename(self.view.left_im.source)

        self.display_parameters_on_logs(
            image = left_filename,
            parameters = parameters,
            values = values
        )

        new_row = {f"Ex_{k}": round(v*100, 2) for k,v in zip(parameters, values)}

        parameter = self.view.parameter_dropdown_item.text
        if parameter == 'DBH':
            results_file = os.path.join(self.RESULTS_DIR, f'results_{self.THIS_PROJECT}_dbh.csv')
        elif parameter == 'CD & TH':
            results_file = os.path.join(self.RESULTS_DIR, f'results_{self.THIS_PROJECT}_cd_th.csv')

        results_df = pd.read_csv(results_file, index_col='Filename')
        results_df.loc[left_filename] = new_row
        results_df.to_csv(results_file)

        if self.image_index < len(left_ims) - 1:
            self.image_index += 1
        else:
            self.create_log_widget(text = 'Batch extraction complete', color=(0,1,0,1))
            self.unschedule_batch_extraction()
            self.view.ids.analyse_btn.disabled = False
    


    def display_parameters_on_logs(self, image, parameters, values):
        '''
        Displays the extracted parameters in the logging section
        
        @param image: The left image of stereo pair from which the parameters are extracted
        @param parameters: The parameter(s) being extracted
        @param values: The extracted value of the parameter
        '''
        label_text = f'''=================================================== \n\nImage: {os.path.basename(image)} \n{parameters}: {[round(value, 2) for value in values]}'''
        self.create_log_widget(label_text)



    def update_on_batch_extract(self):
        '''
        Schedules the 'on_batch_extract_function' to run every 500ms
        '''
        if not self.verify_config_file():
            self.create_log_widget(text = "Missing camera configuration file path!", color=(1,0,0,1))
            self.unschedule_batch_extraction()
        
        elif not self.verify_project_name():
            self.create_log_widget(text = "Missing the project name! Please provide one to proceed.", color=(1,0,0,1))
            self.unschedule_batch_extraction()
        
        else:
            Clock.schedule_interval(self.on_batch_extract, 0.5)
    


    def unschedule_batch_extraction(self):
        '''
        Unschedules the 'on_batch_extract_function' to stop it once batch extraction is complete
        '''
        Clock.unschedule(self.on_batch_extract)

    

    def compute_parameter(self, mask_path):
        '''
        Computes the parameter selected by the user e.g. DBH, CD, TH. Both CD and TH are computed at once since they are
        both extracted from the same segmented disparity map.

        @param mask_path: The path where the segmented disparity map is saved
        '''
        parameter = self.view.parameter_dropdown_item.text
        dmap = cv2.imread(self.view.right_im.source, 0)
        mask = cv2.imread(mask_path, 0)
        mask = algorithms.rectify(mask, self.CONFIG_FILE_PATH, "left")
        K, _, _, _, _, _, _, _, T, _ = algorithms.load_camera_params(self.CONFIG_FILE_PATH)

        focal_length = K[0, 0]
        cx = K[0, 2]
        cy = K[1, 2]
        baseline = T[0, 0] / 1000
        
        if parameter == "DBH":
            inputs = {
                "image": dmap,
                "mask": mask, 
                "baseline": baseline,
                "focal_length": focal_length,
                "dfov": self.DIAG_FIELD_OF_VIEW,
                "cx": cx,
                "cy": cy
            }
            return [[parameter], [algorithms.compute_dbh(**inputs)]]
        
        elif parameter == "CD & TH":
            parameters = ["CD", "TH"]
            inputs = {
                "image": dmap,
                "baseline": baseline,
                "focal_length": focal_length, 
                "dfov": self.DIAG_FIELD_OF_VIEW,
                "cx": cx,
                "cy": cy
            }
            values = [algorithms.compute_cd(**inputs), algorithms.compute_th(**inputs)]

            return [parameters, values]
    


    def analyse_results(self):
        '''
        Analyses the extracted results by comparing them to the ground truth values. It also shows
        regression plots for all the three parameters
        '''
        parameter = self.view.parameter_dropdown_item.text
        if parameter == 'DBH':
            file_path = os.path.join(self.RESULTS_DIR, f'results_{self.THIS_PROJECT}_dbh.csv')
        elif parameter == 'CD & TH':
            file_path = os.path.join(self.RESULTS_DIR, f'results_{self.THIS_PROJECT}_cd_th.csv')
        
        if self.REF_PARAMS_FILE:
            df = pd.read_csv(file_path, index_col='Filename')
            df2 = pd.read_csv(self.REF_PARAMS_FILE, index_col='Filename')
            

            if parameter == 'CD & TH':
                df['Ref_TH'] = df2['Ref_TH']
                df['Ref_CD'] = df2['Ref_CD']
                df['AE_TH (cm)'] = round((df['Ref_TH'] - df['Ex_TH']).abs(), 2)
                df['APE_TH (%)'] = round((df['AE_TH (cm)'] / df['Ref_TH']) * 100, 2)
                df['AE_CD (cm)'] = round((df['Ref_CD'] - df['Ex_CD']).abs(), 2)
                df['APE_CD (%)'] = round((df['AE_CD (cm)'] / df['Ref_CD']) * 100, 2)

                cd_mae = df['AE_CD (cm)'].mean()
                cd_mape = df['APE_CD (%)'].mean()
                cd_rmse = np.sqrt(mean_squared_error(df['Ref_CD'], df['Ex_CD']))
                th_mae = df['AE_TH (cm)'].mean()
                th_mape = df['APE_TH (%)'].mean()
                th_rmse = np.sqrt(mean_squared_error(df['Ref_TH'], df['Ex_TH']))
                
                df.to_csv(file_path)
                
                text = f'''\n\n\n\n\nAnalysis of CD & TH results Complete...\n\nMAE_CD: {round(cd_mae, 2)} cm \nMAPE_CD: {round(cd_mape, 2)} % \nRMSE_CD: {round(cd_rmse, 2)} cm \n\nMAE_TH: {round(th_mae, 2)} cm \nMAPE_TH: {round(th_mape, 2)} % \nRMSE_TH: {round(th_rmse, 2)} cm \n\n\n\nResults saved to {file_path}'''
                
                self.create_log_widget(
                    text = text,
                    color = (0,1,0,1)
                )
                self.plot_regression(
                    parameter = 'CD',
                    x = df['Ref_CD'],
                    y = df['Ex_CD'],
                    path = os.path.join(self.RESULTS_DIR, 'regression_CD.jpg')
                )
                self.plot_regression(
                    parameter = 'TH',
                    x = df['Ref_TH'],
                    y = df['Ex_TH'],
                    path = os.path.join(self.RESULTS_DIR, 'regression_TH.jpg')
                )
                self.view.ids.left_im.source = os.path.join(self.RESULTS_DIR, 'regression_CD.jpg')
                self.view.ids.right_im.source = os.path.join(self.RESULTS_DIR, 'regression_TH.jpg')
                self.create_log_widget(
                    text = '\n\n\n\nRegression plot generation complete...',
                    color = (0,1,0,1)
                )
            
            elif parameter == 'DBH':
                df['Ref_DBH'] = df2['Ref_DBH']
                df['AE_DBH (cm)'] = round((df['Ref_DBH'] - df['Ex_DBH']).abs(), 2)
                df['APE_DBH (%)'] = round((df['AE_DBH (cm)'] / df['Ref_DBH']) * 100, 2)

                dbh_mae = df['AE_DBH (cm)'].mean()
                dbh_mape = df['APE_DBH (%)'].mean()
                dbh_rmse = np.sqrt(mean_squared_error(df['Ref_DBH'], df['Ex_DBH']))

                df.to_csv(file_path)

                text = f'''Analysis of DBH results Complete...\n\nMAE_DBH: {round(dbh_mae, 2)} cm \nMAPE_DBH: {round(dbh_mape, 2)} % \nRMSE_DBH: {round(dbh_rmse, 2)} cm \n\nResults saved to {file_path}'''

                self.create_log_widget(
                    text = text,
                    color = (0,1,0,1)
                )
                self.plot_regression(
                    parameter = 'DBH',
                    x = df['Ref_DBH'],
                    y = df['Ex_DBH'],
                    path = os.path.join(self.RESULTS_DIR, 'regression_DBH.jpg')
                )
                self.view.ids.right_im.source = os.path.join(self.RESULTS_DIR, 'regression_DBH.jpg')
                self.create_log_widget(
                    text = '\n\n\n\nRegression plot generation complete...',
                    color = (0,1,0,1)
                )
        else:
            self.create_log_widget(
                text = '\n\n\nMissing the reference parameters file.',
                color = (1,0,0,1)
            )

    


    def plot_regression(self, parameter, x, y, path):
        '''
        Plots a regression line and saves it to the 'path'
        @param parameter: Parameter of interest e.g. DBH
        @param x: Arraylike
        @param y: Arraylike
        @param path: Path where figure will be saved
        '''
        
        xmin = np.floor(x.min())
        xmax = np.ceil(x.max())
        ymin = np.floor(y.min())
        ymax = np.ceil(y.max())

        m, c = np.polyfit(x, y, 1)

        plt.figure()
        plt.grid()
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.locator_params(tight=True, nbins=7)
        plt.minorticks_on()
        plt.scatter(x, y, color='blue', s=10)
        plt.plot(np.linspace(xmin, xmax, 1000), np.linspace(ymin, ymax, 1000), color='green', linestyle='dashed', label='1:1 line')
        plt.plot(x, m*x + c, color='red', label='regression line')
        plt.title(f'Regression Plot for {parameter}')
        plt.xlabel(f'Reference {parameter} Values (cm)')
        plt.ylabel(f'Extracted {parameter} Values (cm)') 
        plt.legend()
        plt.savefig(path, dpi=600)
    


    def show_confirmation_dialog(self):
        '''
        Shows a popup dialog modal for the user to confirm that they want the app settings to be reset. \n
        Called when the reset button is pressed in the user interface
        '''
        if not self.dialog:
            self.dialog = MDDialog(
                title="Reset app settings",
                type="custom",
                content_cls=RefreshConfirm(),
                auto_dismiss = False,
                buttons=[
                    MDRaisedButton(
                        text="CANCEL",
                        theme_text_color="Custom",
                        text_color="white",
                        md_bg_color="red",
                        on_release=self.close_confirmation_dialog,
                    ),
                    MDRaisedButton(
                        text="CONTINUE",
                        theme_text_color="Custom",
                        text_color="white",
                        md_bg_color="green",
                        on_release=self.reset,
                    )
                ],
            )
        self.dialog.open()
    


    def show_info(self):
        '''
        Called when the user clicks on the 'About' button in the user interface. It displays a popup modal with
        information about the TreeVision software
        '''
        if not self.info_popup_modal:
            self.info_popup_modal = MDDialog(
                title="About TreeVision",
                type="custom",
                content_cls = InfoPopupModal(),
                auto_dismiss=False,
                buttons=[
                    MDRaisedButton(
                        text="OK",
                        theme_text_color="Custom",
                        text_color="white",
                        md_bg_color="green",
                        on_release=self.close_info_popup,
                    )
                ]
            )
        self.info_popup_modal.open()
    


    def close_confirmation_dialog(self, instance):
        '''
        Dismisses the confirmation popup dialog
        '''
        self.dialog.dismiss()
    


    def close_info_popup(self, instance):
        '''
        Dismisses the app info popup modal
        '''
        self.info_popup_modal.dismiss()
    

    def open_user_guide(self):
        '''
        Opens the User Guide of TreeVision using the the system default application
        '''
        os.startfile("TreeVision User Guide.pdf")    
    
    
    def reset(self, instance):
        '''
        Resets all configuration variables to their default values and resets the app in readiness to begin a fresh extraction
        '''
        self.image_index = 0
        self.num_of_images = 0
        self.DISPARITY_MAPS_DIR = None
        self.RESULTS_DIR = None
        self.IMAGES_DIR = None
        self.FILE_MANAGER_SELECTOR = 'folder'
        self.CONFIG_FILE_PATH = None

        self.set_display_images()
        self.toggle_scrolling_icons()
        self.initialize_sgbm_values()

        self.view.ids.project_name.text = 'test'
        self.view.ids.parameter_dropdown_item.text = 'Select parameter'
        self.view.ids.segmentation_dropdown_item.text = 'Select approach'
        self.view.ids.rectification_dropdown_item.text = 'No'
        self.view.ids.dfov.text = '55'
        self.view.ids.analyse_btn.disabled = True
        self.view.ids.batch_extract_btn.disabled = True
        self.view.ids.scroll_layout.clear_widgets()

        label_text = "App has been reset and all configurations cleared."
        self.dialog.dismiss()
        self.create_log_widget(label_text)
    


    def create_log_widget(self, text, color=(1,1,1,1)):
        '''
        Creates a widget to be added to the logging section on the user interfac
        @param text: The text contained on the widget
        '''
        logwidget = MDLabel(
                text = text,
                text_size = (None, None),
                valign = 'middle',
                theme_text_color = "Custom",
                text_color = color
            )
        
        layout = self.view.ids.scroll_layout
        scrollview = self.view.ids.scrollview
        layout.add_widget(logwidget)
        scrollview.scroll_y = 0


class IconListItem(OneLineIconListItem):
    icon = StringProperty()