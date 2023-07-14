import os
import time
import importlib
import csv
import matplotlib.pyplot as plt
from glob import glob

from kivy.core.window import Window
from kivy.metrics import dp
from kivy.utils import rgba
from kivy.properties import StringProperty

from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.list import OneLineIconListItem
from kivymd.uix.button import MDIconButton
from kivymd.uix.label import MDLabel
from kivymd.toast import toast
from kivy.clock import Clock

import View.ExtractScreen.extract_screen
from . import algorithms
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

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
    num_of_images = 0

    ASSET_DIR = 'assets'
    PROJECT_DIR = os.path.join(ASSET_DIR, 'projects')
    DISPARITY_MAPS_DIR = None
    RESULTS_DIR = None
    IMAGES_DIR = 'test/full_trees'
    FILE_MANAGER_SELECTOR = 'folder'
    SELECT_BUTTON_ID = None
    CONFIG_FILE_PATH = 'configs/stereo_sample.yml'
    REF_PARAMS_FILE = None

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
        self.initialize_sgbm_values()

    

    def toggle_scrolling_icons(self):
        '''
        Toggles the buttons for scrolling the images left and right based on whether the project path has been selected
        or not. The buttons are toggled off if a project path with multiple images has not been selected.
        '''

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

        self.file_manager.show(os.path.expanduser("."))
        self.manager_open = True

    

    def select_path(self, path: str):
        '''
        It will be called when you click on the file name
        or the catalog selection button.

        @param path: path to the selected directory or file;
        '''

        if self.FILE_MANAGER_SELECTOR == 'folder':
            self.IMAGES_DIR = path 
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
        
        left_ims = glob(os.path.join(self.IMAGES_DIR, '*LEFT.jpg'))
        right_ims = glob(os.path.join(self.IMAGES_DIR, '*RIGHT.jpg'))

        self.num_of_images = len(left_ims)
        self.view.ids.progress_bar.max = self.num_of_images

        if self.verify_images(left_ims, right_ims):
            return (left_ims, right_ims)
        
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


        left, right = self.load_stereo_images()
        left = sorted(left)
        right = sorted(right)

        if self.on_button_press(button_id):
            self.view.left_im.source = left[self.image_index]
            self.view.right_im.source = right[self.image_index]



    def create_project_directories(self):
        '''
        Creates a directory in the "assets" folder of the app for the project. This is the directory where the extracted disparity maps as well as a CSV file containing the extracted parameters will be saved
        '''

        project = self.view.project_name.text 
        
        project_path = os.path.join(self.PROJECT_DIR, f'{project}')
        dmaps_path = os.path.join(project_path, 'disparity_maps')
        results_path = os.path.join(project_path, 'results')

        self.DISPARITY_MAPS_DIR = dmaps_path
        self.RESULTS_DIR = results_path
        
        if not os.path.exists(project_path):
            os.makedirs(dmaps_path) if not os.path.exists(dmaps_path) else None
            os.makedirs(results_path) if not os.path.exists(results_path) else None
            self.create_log_widget(text = "Project folders have been created!")

        parameter = self.view.parameter_dropdown_item.text
        if parameter == 'DBH':
            results_file = os.path.join(self.RESULTS_DIR, 'results_dbh.csv')
            columns = ['Ref_DBH', 'Ex_DBH', 'AE_DBH (cm)', 'APE_DBH (%)']
        elif parameter == 'CD & TH':
            results_file = os.path.join(self.RESULTS_DIR, 'results_cd_th.csv')
            columns = ['Ref_TH', 'Ex_TH', 'AE_TH (cm)', 'APE_TH (%)', 'Ref_CD', 'Ex_CD', 'AE_CD (cm)', 'APE_CD (%)']

        if not os.path.exists(results_file):
            results_df = pd.DataFrame(columns=columns)
            results_df.index.name = 'Filename'
            results_df.to_csv(results_file)
        
        return True



    def compute_and_save_disparity(self):
        '''
        Saves the extracted disparity map in the project folder and displays it in the user interface on the position initially occupied by the right image.
        It returns the paths to the segmentation mask and the segmented disparity map
        '''

        left_img_path = self.view.left_im.source
        right_img_path = self.view.right_im.source

        folder_path = os.path.dirname(left_img_path)
        left_img_filename = os.path.basename(left_img_path)
        mask_filename = left_img_filename.split(".")[0] + "_mask.png"

        mask_path =  os.path.join(folder_path, mask_filename)
        print(f"MASK: {mask_path}")

        left = cv2.imread(left_img_path, 0)
        right = cv2.imread(right_img_path, 0)
        mask = cv2.imread(mask_path, 0)
        kernel = np.ones((3,3), np.uint8)

        if self.verify_config_file():
            dmap = algorithms.extract(
                left_im = left, 
                right_im = right, 
                mask = mask, 
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
            
            dmap_filename = left_img_path.split('\\')[-1].split('.')[0] + '_disparity.jpg'
            dmap_path = os.path.join(self.DISPARITY_MAPS_DIR, dmap_filename)

            cv2.imwrite(dmap_path, dmap)
            return dmap_path, mask_path
        else:
            return False
    


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
            dmap_path, mask_path = self.compute_and_save_disparity()
            self.view.right_im.source = dmap_path

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
                results_file = os.path.join(self.RESULTS_DIR, 'results_dbh.csv')
            elif parameter == 'CD & TH':
                results_file = os.path.join(self.RESULTS_DIR, 'results_cd_th.csv')

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
        
        left_ims, right_ims = self.load_stereo_images()
        left_ims = sorted(left_ims)
        right_ims = sorted(right_ims)

        left_img = left_ims[self.image_index]
        right_img = right_ims[self.image_index]

        self.view.left_im.source = left_img
        self.view.right_im.source = right_img

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
            results_file = os.path.join(self.RESULTS_DIR, 'results_dbh.csv')
        elif parameter == 'CD & TH':
            results_file = os.path.join(self.RESULTS_DIR, 'results_cd_th.csv')

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
        
        if parameter == "DBH":
            return [[parameter], [algorithms.compute_dbh(dmap, mask)]]
        
        elif parameter == "CD & TH":
            parameters = ["CD", "TH"]
            values = [algorithms.compute_cd(dmap), algorithms.compute_th(dmap)]

            return [parameters, values]
    


    def analyse_results(self):
        '''
        Analyses the extracted results by comparing them to the ground truth values. It also shows
        regression plots for all the three parameters
        '''
        parameter = self.view.parameter_dropdown_item.text
        if parameter == 'DBH':
            file_path = os.path.join(self.RESULTS_DIR, 'results_dbh.csv')
        elif parameter == 'CD & TH':
            file_path = os.path.join(self.RESULTS_DIR, 'results_cd_th.csv')
        
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
            
            text = f'''\n\n\n\nAnalysis of CD & TH results Complete...\n\nMAE_CD: {round(cd_mae, 2)} cm \nMAPE_CD: {round(cd_mape, 2)} % \nRMSE_CD: {round(cd_rmse, 2)} cm \n\nMAE_TH: {round(th_mae, 2)} cm \nMAPE_TH: {round(th_mape, 2)} % \nRMSE_TH: {round(th_rmse, 2)} cm \n\nResults saved to {file_path}'''
            
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
                text = '\n\n\n\n\nRegression plot generation complete...',
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
                text = '\n\n\nRegression plot generation complete...',
                color = (0,1,0,1)
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
    
    
    
    def reset(self):
        '''
        Clears all configuration variables and resets the app in readiness to begin a fresh extraction
        '''
        self.image_index = 0
        self.num_of_images = 0
        self.DISPARITY_MAPS_DIR = None
        self.RESULTS_DIR = None
        self.IMAGES_DIR = None
        self.FILE_MANAGER_SELECTOR = 'folder'
        self.CONFIG_FILE_PATH = None

        self.toggle_scrolling_icons()
        self.initialize_sgbm_values()

        self.view.ids.project_name.text = ''
        self.view.ids.parameter_dropdown_item.text = 'Select parameter'
        self.view.ids.segmentation_dropdown_item.text = 'Select approach'
        self.view.ids.scroll_layout.clear_widgets()

        label_text = "App has been reset and all configurations cleared."
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