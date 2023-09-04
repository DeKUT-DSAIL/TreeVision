import importlib

import View.CalibrateScreen.calibrate_screen
from View.CalibrateScreen.calibrate_screen import RefreshConfirm

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
from sys import platform
from Controller.utils import utils

from kivy.metrics import dp
from kivy.clock import Clock
from kivy.factory import Factory
from kivy.uix.textinput import TextInput
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.label import MDLabel
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.dialog import MDDialog

class CalibrateScreenController:
    """
    The `CalibrateScreenController` class represents a controller implementation.
    Coordinates work of the view with the model.
    The controller implements the strategy pattern. The controller connects to
    the view to control its actions.
    """

    dialog = None

    FILE_MANAGER_SELECTOR = 'folder'
    BUTTON_ID = None
    PROJECT_DIR = os.path.join('assets', 'projects')
    ASSET_IMS_DIR = os.path.join('assets', 'images')
    CONFIGS_DIR = "configs"
    IMAGES_DIR = None
    LEFT_CONFIG_FILE = None
    RIGHT_CONFIG_FILE = None
    SQUARE_SIZE = None

    LEFT_IMS = None
    RIGHT_IMS = None
    UNPAIRED_IMS = None

    image_index = 0
    num_of_images = 0
    objpoints = []
    imgpoints = []
    left_imgpoints = []
    right_imgpoints = []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def __init__(self):

        self.view = View.CalibrateScreen.calibrate_screen.CalibrateScreenView(controller=self)

        self.file_manager = MDFileManager(
            selector = self.FILE_MANAGER_SELECTOR,
            exit_manager = self.exit_manager,
            select_path = self.select_path
        )

        self.set_display_images()
        self.toggle_scrolling_icons()

    def get_view(self) -> View.CalibrateScreen.calibrate_screen:
        return self.view
    


    def set_display_images(self):
        '''
        Sets the display images
        '''
        if platform == "win32":
            self.view.ids.left_image.source = "assets\\images\\calibration\\pattern_cropped.png"
            self.view.ids.right_image.source = "assets\\images\\calibration\\pattern_cropped.png"
        elif platform in ["linux", "linux2"]:
            self.view.ids.left_image.source = "assets/images/calibration/pattern_cropped.png"
            self.view.ids.right_image.source = "assets/images/calibration/pattern_cropped.png"
    


    def set_item(self, menu, dropdown_item, text_item):
        dropdown_item.set_item(text_item)
        dropdown_item.text = text_item
        menu.dismiss()
    


    def file_manager_open(self, selector, button_id):
        '''
        Opens the file manager when the triggering event in the user interface happens
        '''
        self.FILE_MANAGER_SELECTOR = selector
        self.BUTTON_ID = button_id

        self.file_manager = MDFileManager(
            selector = self.FILE_MANAGER_SELECTOR,
            exit_manager = self.exit_manager,
            select_path = self.select_path
        )

        self.file_manager.show(os.path.expanduser("."))
        self.manager_open = True
    


    def select_path(self, path: str):
        '''
        It will be called when you click on the file name or the catalog selection button.

        @param path: path to the selected directory or file;
        '''

        if self.FILE_MANAGER_SELECTOR == 'folder':
            self.IMAGES_DIR = path 
            self.load_images()
            self.create_log_widget(f"Calibration images directory has been selected.\nIMAGES DIRECTORY PATH: {path}")
        else:
            if self.BUTTON_ID == "left":
                self.LEFT_CONFIG_FILE = path
                self.create_log_widget(f"Left camera configuration file has been selected.\nFILE PATH: {path}")
            elif self.BUTTON_ID == "right":
                self.RIGHT_CONFIG_FILE = path
                self.create_log_widget(f"Right camera configuration file has been selected.\nFILE PATH: {path}")
        
        if self.verify_config_files():
            self.view.ids.calibrate_stereo.disabled = False
        
        self.toggle_scrolling_icons()
        self.exit_manager()

    

    def exit_manager(self, *args):
        '''Called when the user reaches the root of the directory tree.'''

        self.manager_open = False
        self.file_manager.close()

    

    def toggle_scrolling_icons(self):
        '''
        Toggles the buttons for scrolling the images left and right based on whether the project images have been loaded
        or not. The buttons are toggled off if a project path with multiple images has not been selected.
        '''

        if self.LEFT_IMS == None and self.RIGHT_IMS == None and self.UNPAIRED_IMS == None:
            self.view.ids.previous_arrow.opacity = 0
            self.view.ids.next_arrow.opacity = 0
        
        elif len(self.LEFT_IMS) > 0 or len(self.RIGHT_IMS) > 0 or len(self.UNPAIRED_IMS) > 0:
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

        left, right, unpaired = self.LEFT_IMS, self.RIGHT_IMS, self.UNPAIRED_IMS
        left = sorted(left)
        right = sorted(right)
        unpaired = sorted(unpaired)

        if self.on_button_press(button_id):
            if len(left) > 0 and len(right) > 0:
                self.view.ids.left_image.source = left[self.image_index]
                self.view.ids.right_image.source = right[self.image_index]
            elif len(unpaired) > 0:
                self.view.ids.left_image.source = unpaired[self.image_index]
    


    def verify_images(self, left_ims, right_ims):
        return len(left_ims) == len(right_ims)
    


    def verify_config_files(self):
        return self.LEFT_CONFIG_FILE != None and self.RIGHT_CONFIG_FILE != None
    
    
    
    
    def load_images(self):
        '''
        Returns the paths to the calibration images. This works for both stereo and single camera calibration
        '''
        
        left_patterns = ['*LEFT*.jpg', '*left*.jpg', '*LEFT*.png', '*left*.png']
        right_patterns = ['*RIGHT*.jpg', '*right*.jpg', '*RIGHT*.png', '*right*.png']
        unpaired_patterns = ['*.jpg', '*.png']

        left_ims = []
        right_ims = []
        unpaired_ims = []

        for pattern in left_patterns:
            left_ims.extend(glob(os.path.join(self.IMAGES_DIR, pattern)))
        
        for pattern in right_patterns:
            right_ims.extend(glob(os.path.join(self.IMAGES_DIR, pattern)))
        
        left_ims = sorted(list(set(left_ims)))
        right_ims = sorted(list(set(right_ims)))
        self.LEFT_IMS = left_ims
        self.RIGHT_IMS = right_ims
        
        if (len(self.RIGHT_IMS) > 0) != (len(self.LEFT_IMS) > 0):
            for pattern in unpaired_patterns:
                unpaired_ims.extend(glob(os.path.join(self.IMAGES_DIR, pattern)))
        
        unpaired_ims = sorted(list(set(unpaired_ims)))
        self.UNPAIRED_IMS = unpaired_ims
        # unpaired_ims = self.LEFT_IMS if self.LEFT_IMS != None else self.RIGHT_IMS
        # self.UNPAIRED_IMS = unpaired_ims
        
        if len(unpaired_ims) > 0:
            self.num_of_images = len(unpaired_ims)
            self.view.ids.progress_bar.max = self.num_of_images
            self.view.ids.left_image.source = unpaired_ims[0]
            self.view.ids.calibrate_single.disabled = False
            self.create_log_widget(
                text = "You have successfully loaded unpaired images...",
                color = (0,1,0,1)
            )

        elif len(self.RIGHT_IMS) > 0 and len(self.LEFT_IMS) > 0:
            if self.verify_images(left_ims, right_ims):
                self.num_of_images = min(len(left_ims), len(right_ims))
                self.view.ids.progress_bar.max = self.num_of_images
                self.view.ids.left_image.source = left_ims[0]
                self.view.ids.right_image.source = right_ims[0]
                self.create_log_widget(
                    text = "You have successfully loaded stereo images...",
                    color = (0,1,0,1)
                )
            else:
                self.create_log_widget(
                    text = "Number of Left and Right Images NOT equal!",
                    color = (1,0,0,1)
                )
    


    def save_points(self, dt):
        """
        Saves object points and image points obtained from calibration image
        """

        if len(self.RIGHT_IMS) > 0 and len(self.LEFT_IMS) > 0:
            self.create_log_widget(
                text = "You seem to have paired LEFT and RIGHT images in this directory! Please remove any paired images before proceeding...",
                color = (1,0,0,1)
            )
            self.unschedule_on_calibrate()

        images = self.UNPAIRED_IMS
        self.view.ids.left_image.source = images[self.image_index]

        image = self.view.ids.left_image.source
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        height = int(self.view.ids.pattern_height.text)
        width = int(self.view.ids.pattern_width.text)

        objp = np.zeros((height*width, 3), np.float32)
        objp[:,:2] = np.mgrid[0:width, 0:height].T.reshape(-1,2)

        self.SQUARE_SIZE = float(self.view.ids.square_size.text)
        objp = objp * self.SQUARE_SIZE

        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        if ret == True:
            self.objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), self.criteria)
            self.imgpoints.append(corners2)
            
            cv2.drawChessboardCorners(img, (width, height), corners2, ret)
            drawn = os.path.join(self.ASSET_IMS_DIR, f'calibration/drawn_{self.image_index}.jpg')
            cv2.imwrite(drawn, img)
            self.view.ids.right_image.source = drawn

        self.view.ids.progress_bar.value = self.image_index + 1
        
        if self.image_index < len(images) - 1:
            self.image_index += 1
        else:
            self.create_log_widget(text = 'Object and Image Points Saved...')
            self.on_calibrate()
            drawn_files = glob(os.path.join(self.ASSET_IMS_DIR, 'calibration/drawn*.jpg'))
            for file in drawn_files:
                os.remove(file)
            self.unschedule_on_calibrate()
    


    def save_stereo_points(self, dt):
        """
        Saves object points and image points obtained from stereo images
        """

        left, right = self.LEFT_IMS, self.RIGHT_IMS

        if not self.verify_images(left, right):
            self.create_log_widget(
                    text = "Number of Left and Right Images NOT equal! \nPlease check before proceeding...",
                    color = (1,0,0,1)
                )
            self.unschedule_save_stereo_points()
        
        self.view.ids.left_image.source = left[self.image_index]
        self.view.ids.right_image.source = right[self.image_index]

        imageL = self.view.ids.left_image.source
        imageR = self.view.ids.right_image.source

        imgL = cv2.imread(imageL)
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)

        imgR = cv2.imread(imageR)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        height = int(self.view.ids.pattern_height.text)
        width = int(self.view.ids.pattern_width.text)

        objp = np.zeros((height*width, 3), np.float32)
        objp[:,:2] = np.mgrid[0:width, 0:height].T.reshape(-1,2)

        self.SQUARE_SIZE = float(self.view.ids.square_size.text)
        objp = objp * self.SQUARE_SIZE

        ret_left, corners_left = cv2.findChessboardCorners(grayL, (width, height), None)
        ret_right, corners_right = cv2.findChessboardCorners(grayR, (width, height), None)

        if ret_left and ret_right:
            self.objpoints.append(objp)
            
            corners2_left = cv2.cornerSubPix(grayL, corners_left, (11,11), (-1,-1), self.criteria)
            self.left_imgpoints.append(corners2_left)

            corners2_right = cv2.cornerSubPix(grayR, corners_right, (11,11), (-1,-1), self.criteria)
            self.right_imgpoints.append(corners2_right)
            
            cv2.drawChessboardCorners(imgL, (width, height), corners2_left, ret_left)
            cv2.drawChessboardCorners(imgR, (width, height), corners2_right, ret_right)

            drawn_left = os.path.join(self.ASSET_IMS_DIR, f'calibration/drawn_left_{self.image_index}.jpg')
            drawn_right = os.path.join(self.ASSET_IMS_DIR, f'calibration/drawn_right_{self.image_index}.jpg')

            cv2.imwrite(drawn_left, imgL)
            cv2.imwrite(drawn_right, imgR)

            self.view.ids.left_image.source = drawn_left
            self.view.ids.right_image.source = drawn_right
        else:
            self.create_log_widget(text = f"Couldn't find chessboard corners for {os.path.basename(imageL)} and {os.path.basename(imageR)}", color = (1,0,0,1))
            self.unschedule_save_stereo_points()

        self.view.ids.progress_bar.value = self.image_index + 1
        
        if self.image_index < len(left) - 1:
            self.image_index += 1
        else:
            self.create_log_widget(text = 'Object Points and Stereo Image Points Saved...')
            self.stereo_calibrate()
            drawn_files = glob(os.path.join(self.ASSET_IMS_DIR, 'calibration/drawn*.jpg'))
            for file in drawn_files:
                os.remove(file)
            self.unschedule_save_stereo_points()



    def single_calibrate(self):
        '''
        Performs calibration of a single camera 
        '''

        x = int(self.view.ids.image_height.text)
        y = int(self.view.ids.image_width.text)

        flags = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_ZERO_TANGENT_DIST 

        mtx_init = np.array([[1500, 0, 640], [0, 1500, 360], [0, 0, 1]], dtype=np.int16)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objectPoints = self.objpoints,
            imagePoints = self.imgpoints, 
            imageSize = (x, y),
            cameraMatrix = mtx_init,
            distCoeffs = None,
            flags = flags
        )

        return [ret, mtx, dist, rvecs, tvecs, self.imgpoints, self.objpoints]



    def stereo_calibrate(self):
        '''
        Performs calibration of a stereo camera
        '''

        project_name = self.view.ids.project_name.text
        project_dir_path = os.path.join(self.CONFIGS_DIR, project_name)
        if not os.path.exists(project_dir_path):
            os.makedirs(project_dir_path)

        stereo_save_file = os.path.join(project_dir_path, f"{self.view.ids.save_file.text}.yml")

        K1, D1 = utils.load_coefficients(self.LEFT_CONFIG_FILE)
        K2, D2 = utils.load_coefficients(self.RIGHT_CONFIG_FILE)

        h = int(self.view.ids.image_height.text)
        w = int(self.view.ids.image_width.text)

        ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
            objectPoints = self.objpoints, 
            imagePoints1 = self.left_imgpoints, 
            imagePoints2= self.right_imgpoints, 
            cameraMatrix1 = K1, 
            distCoeffs1 = D1, 
            cameraMatrix2 = K2, 
            distCoeffs2 = D2, 
            imageSize = (h,w)
        )

        R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
            cameraMatrix1 = K1,
            distCoeffs1 = D1,
            cameraMatrix2 = K2,
            distCoeffs2 = D2,
            imageSize = (h, w),
            R = R,
            T = T,
            flags = cv2.CALIB_ZERO_DISPARITY,
            alpha = 0.9
        )

        utils.save_stereo_coefficients(stereo_save_file, K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q)
        self.create_log_widget(text = f"Stereo Calibration Complete \nRMS: {round(ret, 4)}", color = (0,1,0,1))
    


    def on_calibrate(self):
        '''
        Called when the 'Calibrate' button is pressed in the user interface
        '''

        ret, K, D, R, T, image_points, object_points = self.single_calibrate()
        error_info = utils.projection_error(object_points, image_points, T, R, K, D)

        self.create_log_widget(
            text = f"Calibration finished \nCalibration RMS: {round(ret, 4)} \nCalibration ME: {round(error_info['ME'], 4)}",
            color = (0,1,0,1)
        )

        project_name = self.view.ids.project_name.text
        project_dir_path = os.path.join(self.CONFIGS_DIR, project_name)
        if not os.path.exists(project_dir_path):
            os.makedirs(project_dir_path)

        save_file = self.view.ids.save_file.text
        save_file_path = os.path.join(project_dir_path, f"{save_file}.yml")
        utils.save_coefficients(save_file_path, K, D)
        self.create_log_widget(text = f"Calibration parameters saved to: {save_file_path}")

        self.plot_scatter(error_info, project_dir_path)

        scatter_plot_path = os.path.join(project_dir_path, "calib_error_scatter.jpg")
        self.view.ids.right_image.source = scatter_plot_path
        self.create_log_widget(text = "Error Scatter Plot Created", color = (0,1,0,1))
    


    def update_save_points(self):
        '''
        Schedules the 'save_points' function to run every 500ms
        '''
        if self.check_valid_input() and self.check_file_selection('single'):
            Clock.schedule_interval(self.save_points, 0.5)
    


    def update_save_stereo_points(self):
        '''
        Schedules the 'save_stereo_points' function to run every 500ms
        '''
        if self.check_valid_input() and self.check_file_selection('stereo'):
            Clock.schedule_interval(self.save_stereo_points, 0.5)
    


    def unschedule_on_calibrate(self):
        '''
        Unschedules the 'save_points' to stop it once camera calibration is complete
        '''
        Clock.unschedule(self.save_points)
    


    def unschedule_save_stereo_points(self):
        '''
        Unschedules the 'save_stereo_points' to stop it once stereo calibration is complete
        '''
        Clock.unschedule(self.save_stereo_points)



    def plot_scatter(self, error_info, path):
        """
        This function creates a scatterplot of the residual errors due to differences between original and 
        reprojected image points
        
        @param error_info: Error information including the mean error and errors along the X and Y axes
        """
        
        x = error_info['X']
        y = error_info['Y']
        mean_error = error_info['ME']

        labels = ["Image " + str(i) for i in range(self.num_of_images)]
        colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for i in range(6)]) for j in range(self.num_of_images)]
        
        plt.close()
        plt.figure(figsize=(10, 7))
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title("Reprojection Errors in Pixels")

        for i in range(len(colors)):
            plt.scatter(x[i], y[i], marker='+', color=colors[i], label=labels[i])
        
        plt.legend(bbox_to_anchor=(1.05,1.0), loc='upper left')
        plt.tight_layout()
        plt.text(-1.1, 1.7, 'ME' + str(round(mean_error, 4)))

        plt.savefig(os.path.join(path, "calib_error_scatter.jpg"), dpi=600)
    


    def check_valid_input(self):
        '''
        Checks that all required text input is valid
        '''
        inputs = [
            self.view.ids.project_name,
            self.view.ids.save_file,
            self.view.ids.image_width,
            self.view.ids.image_height,
            self.view.ids.pattern_width,
            self.view.ids.pattern_height,
            self.view.ids.square_size
        ]

        checks = [input.is_valid() for input in inputs]

        if not all(checks):
            self.create_log_widget(
                text = f"Missing these inputs: {[input.name for input in inputs if not input.is_valid()]}.",
                color = (1,0,0,1)
            )
        
        return all(checks)



    def check_file_selection(self, mode):
        '''
        Checks that calibration images and camera configuration file have been selected
        '''
        if mode == 'single':
            file_checks = [self.IMAGES_DIR]
            labels = [
                "Calibration images directory not selected"
            ]
        elif mode == 'stereo':
            file_checks = [self.IMAGES_DIR, self.LEFT_CONFIG_FILE, self.RIGHT_CONFIG_FILE]
            labels = [
                "Calibration images directory not selected",
                "Left camera calibration file not selected",
                "Right camera calibration file not selected"
            ]
        
        statuses = [bool(file_check) for file_check in file_checks]

        for i, status in enumerate(statuses):
            if not status:
                self.create_log_widget(
                    text=labels[i],
                    color=(1, 0, 0, 1)
                )
        
        return all(statuses)



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
    


    def close_confirmation_dialog(self, instance):
        '''
        Dismisses the popup modal
        '''
        self.dialog.dismiss()



    def reset(self, instance):
        '''
        Clears all configuration variables and resets the app in readiness to begin a fresh calibration
        '''
        self.IMAGES_DIR = None
        self.LEFT_CONFIG_FILE = None
        self.RIGHT_CONFIG_FILE = None
        self.SQUARE_SIZE = None

        self.image_index = 0
        self.num_of_images = 0
        self.objpoints = []
        self.imgpoints = []
        self.left_imgpoints = []
        self.right_imgpoints = []

        self.set_display_images()
        self.toggle_scrolling_icons()

        self.view.ids.progress_bar.value = 0
        self.view.ids.project_name.text = ''
        self.view.ids.calibrate_single.disabled = True
        self.view.ids.calibrate_stereo.disabled = True

        label_text = "App has been reset and all configurations cleared."

        self.view.ids.scroll_layout.clear_widgets()
        self.dialog.dismiss()
        self.create_log_widget(label_text)

    

    def create_log_widget(self, text, color=(1,1,1,1)):
        '''
        Creates a widget to be added to the logging section on the user interface
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

        # layout.spacing = logwidget.height * 0.8
        layout.add_widget(logwidget)
        scrollview.scroll_y = 0



class RequiredTextInput(TextInput):
    '''
    Implements a TextInput object that requires text input in the field
    '''
    def __init__(self, name = '', **kwargs):
        super(RequiredTextInput, self).__init__(**kwargs)
        self.id = kwargs.get('id', None)
        self.name = name

    def is_valid(self):
        '''
        Checks that the text is not empty
        '''
        return bool(self.text.strip())


Factory.register('RequiredTextInput', cls=RequiredTextInput)