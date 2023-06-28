import importlib

import View.CalibrateScreen.calibrate_screen

# We have to manually reload the view module in order to apply the
# changes made to the code on a subsequent hot reload.
# If you no longer need a hot reload, you can delete this instruction.
importlib.reload(View.CalibrateScreen.calibrate_screen)

import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from Controller.utils import utils

from kivy.metrics import dp
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.label import MDLabel


class CalibrateScreenController:
    """
    The `CalibrateScreenController` class represents a controller implementation.
    Coordinates work of the view with the model.
    The controller implements the strategy pattern. The controller connects to
    the view to control its actions.
    """

    IMAGES_DIR = None
    PROJECT_DIR = os.path.join('assets', 'projects')
    ASSET_IMS_DIR = os.path.join('assets', 'images')
    CONFIGS_DIR = "configs"

    images = None
    image_index = 0
    num_of_images = 0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def __init__(self):

        self.view = View.CalibrateScreen.calibrate_screen.CalibrateScreenView(controller=self)

        self.calib_type_dropdown_items = [
            {
                "viewclass": "OneLineListItem",
                "text": "Single Camera",
                "height": dp(56),
                "on_release": lambda x="Single Camera": self.set_item(self.calib_type_menu, self.view.ids.calib_type_dropdown_item, x),
            },
            {
                "viewclass": "OneLineListItem",
                "text": "Stereo Camera",
                "height": dp(56),
                "on_release": lambda x="Stereo Camera": self.set_item(self.calib_type_menu, self.view.ids.calib_type_dropdown_item, x),
            }
        ]

        self.calib_type_menu = MDDropdownMenu(
            caller = self.view.ids.calib_type_dropdown_item,
            items = self.calib_type_dropdown_items,
            position = "center",
            background_color = 'brown',
            width_mult = 3,
        )

        self.file_manager = MDFileManager(
            selector = 'folder',
            exit_manager = self.exit_manager,
            select_path = self.select_path
        )

        self.calib_type_menu.bind()
        self.toggle_scrolling_icons()

    def get_view(self) -> View.CalibrateScreen.calibrate_screen:
        return self.view
    


    def set_item(self, menu, dropdown_item, text_item):
        dropdown_item.set_item(text_item)
        dropdown_item.text = text_item
        menu.dismiss()
    


    def file_manager_open(self, selector):
        '''
        Opens the file manager when the triggering event in the user interface happens
        '''

        self.file_manager.show(os.path.expanduser("."))
        self.manager_open = True
    


    def select_path(self, path: str):
        '''
        It will be called when you click on the file name or the catalog selection button.

        @param path: path to the selected directory or file;
        '''

        self.IMAGES_DIR = path 
        self.create_log_widget(f"Project images directory has been selected.\nIMAGES DIRECTORY PATH: {path}")
        
        self.toggle_scrolling_icons()
        self.exit_manager()

    

    def exit_manager(self, *args):
        '''Called when the user reaches the root of the directory tree.'''

        self.manager_open = False
        self.file_manager.close()

    

    def toggle_scrolling_icons(self):
        '''
        Toggles the buttons for scrolling the images left and right based on whether the project path has been selected
        or not. The buttons are toggled off if a project path with multiple images has not been selected.
        '''

        if self.IMAGES_DIR == None:
            self.view.ids.previous_arrow.opacity = 0
            self.view.ids.next_arrow.opacity = 0
        else:
            self.view.ids.previous_arrow.opacity = 1
            self.view.ids.next_arrow.opacity = 1
            self.view.ids.previous_arrow.on_release = lambda: self.show_next_image('previous')
            self.view.ids.next_arrow.on_release = lambda: self.show_next_image('next')
    


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

        left, right = self.load_images()
        left = sorted(left)
        right = sorted(right)

        if self.on_button_press(button_id):
            if len(right) == 0:
                self.view.ids.left_image.source = left[self.image_index]
            else:
                self.view.ids.left_image.source = left[self.image_index]
                self.view.ids.right_image.source = right[self.image_index]
    


    def verify_images(self, left_ims, right_ims):
        return len(left_ims) == len(right_ims)
    
    
    
    def load_images(self):
        '''
        Returns the paths to the calibration images. This works for both stereo and single camera calibration
        '''
        
        left_ims = glob(os.path.join(self.IMAGES_DIR, '*_LEFT.jpg'))
        right_ims = glob(os.path.join(self.IMAGES_DIR, '*_RIGHT.jpg'))

        self.num_of_images = len(left_ims)
        self.view.ids.progress_bar.max = self.num_of_images

        if len(right_ims) == 0:
            self.create_log_widget(text = "SINGLE CALIB: Left Images Loaded")
            return left_ims

        elif self.verify_images(left_ims, right_ims):
            self.create_log_widget(text = "STEREO CALIB: Left and Right Images Loaded")
            return (left_ims, right_ims)
        
        self.create_log_widget(text = "Number of Left and Right Images NOT equal!")
    


    def single_calibrate(self, path, square_size, width, height):
        """
        Calibrates a single camera using calibration images in 'path'

        @param path: Directory where the images are stored
        @param square_size: Size of the square (mm or cm) on the checkerboard pattern
        @param width: Width of the checkerboard pattern (see OpenCV documentation for how to determine dimensions)
        @param height: Height of the checkerboard pattern 
        """

        x = int(self.view.ids.image_height.text)
        y = int(self.view.ids.image_width.text)

        objp = np.zeros((height*width, 3), np.float32)
        objp[:,:2] = np.mgrid[0:width, 0:height].T.reshape(-1,2)

        objp = objp * square_size

        objpoints = [] #3D points in real world space
        imgpoints = [] #2D points in image plane

        self.images = glob(path + '/' + "*_LEFT.jpg")

        for fname in self.images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), self.criteria)
                imgpoints.append(corners2)
                
                cv2.drawChessboardCorners(img, (width, height), corners2, ret)
                drawn = os.path.join(self.ASSET_IMS_DIR, 'calibration/drawn.jpg')
                cv2.imwrite(drawn, img)
                self.view.ids.right_image.source = drawn

            
        flags = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_ZERO_TANGENT_DIST 

        mtx_init = np.array([[1500, 0, 640], [0, 1500, 360], [0, 0, 1]], dtype=np.int16)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objectPoints = objpoints,
            imagePoints = imgpoints, 
            imageSize = (x, y),
            cameraMatrix = mtx_init,
            distCoeffs = None,
            flags = flags
        )

        return [ret, mtx, dist, rvecs, tvecs, imgpoints, objpoints]
    


    def on_calibrate(self):
        '''
        Called when the 'Calibrate' button is pressed in the user interface
        '''

        width = int(self.view.ids.pattern_width.text)
        height = int(self.view.ids.pattern_height.text)
        square_size = float(self.view.ids.square_size.text)

        ret, K, D, R, T, image_points, object_points = self.single_calibrate(
            path = self.IMAGES_DIR,
            square_size = square_size,
            width = width,
            height = height
        )

        self.create_log_widget(text = "Calibration finished")

        save_file = self.view.ids.save_file.text
        save_file_path = os.path.join(self.CONFIGS_DIR, f"{save_file}.yml")
        utils.save_coefficients(save_file_path, K, D)

        error_info = utils.projection_error(object_points, image_points, T, R, K, D)
        self.plot_scatter(error_info)

        scatter_plot_path = os.path.join(self.IMAGES_DIR, "calib_error_scatter.jpg")
        self.view.ids.right_image.source = scatter_plot_path



    def plot_scatter(self, error_info):
        """
        This function creates a scatterplot of the residual errors due to differences between original and 
        reprojected image points
        
        @param objpoints: Object points in the real world
        @param imgpoints: Image point coordinates in the image plane
        @param tvecs: 3 x 1 translation vector obtained during calibration
        @param rvecs: Rotation matrix obtained during calibration
        @param mtx: Camera matrix obtained during calibration
        @param dist: Camera distortion matrix obtained during calibration
        """
        
        x = error_info['X']
        y = error_info['Y']
        mean_error = error_info['ME']

        labels = ["Image " + str(i) for i in range(len(self.images))]
        colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for i in range(6)]) for j in range(len(self.images))]
            
        plt.figure(figsize=(10, 6))
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title("Reprojection Errors in Pixels")

        for i in range(len(colors)):
            plt.scatter(x[i], y[i], marker='+', color=colors[i], label=labels[i])
        
        plt.legend(bbox_to_anchor=(1.05,1.0), loc='upper left')
        plt.tight_layout()
        plt.text(-1.1, 1.7, 'ME' + str(round(mean_error, 4)))

        plt.savefig(os.path.join(self.IMAGES_DIR, "calib_error_scatter.jpg"), dpi=600)
    

    def reset(self):
        '''
        Clears all configuration variables and resets the app in readiness to begin a fresh extraction
        '''
        self.images = None
        self.image_index = 0
        self.num_of_images = 0
        self.IMAGES_DIR = None

        self.toggle_scrolling_icons()

        self.view.ids.project_name.text = ''
        self.view.ids.calib_type_dropdown_item.text = 'Select calibration type'

        label_text = "App has been reset and all configurations cleared."

        self.view.ids.scroll_layout.clear_widgets()
        self.create_log_widget(label_text)

    

    def create_log_widget(self, text):
        '''
        Creates a widget to be added to the logging section on the user interfac
        @param text: The text contained on the widget
        '''
        logwidget = MDLabel(
                text = text,
                text_size = (None, None),
                valign = 'middle',
                theme_text_color = "Custom",
                text_color = (1,1,1,1)
            )
        
        layout = self.view.ids.scroll_layout
        scrollview = self.view.ids.scrollview

        # layout.spacing = logwidget.height * 0.8
        layout.add_widget(logwidget)
        scrollview.scroll_y = 0