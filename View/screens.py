from Controller.calibrate_screen import CalibrateScreenController
from Controller.extract_screen import ExtractScreenController
from Controller.image_screen import ImageScreenController
from Controller.main_screen import MainScreenController
from Controller.upload_screen import UploadScreenController

screens = {
    "calibrate screen": {
        "controller": CalibrateScreenController,
    },

    "extract screen": {
        "controller": ExtractScreenController,
    },

    "main-screen": {
        "controller": MainScreenController,
    },

    "upload-screen": {
        "controller": UploadScreenController,
    },
    
    "image-screen": {
        "controller": ImageScreenController,
    },
}
