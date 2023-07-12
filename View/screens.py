from Controller.capture_screen import CaptureScreenController
from Controller.calibrate_screen import CalibrateScreenController
from Controller.distance_screen import DistanceScreenController
from Controller.extract_screen import ExtractScreenController
from Controller.image_screen import ImageScreenController
from Controller.main_screen import MainScreenController
from Controller.upload_screen import UploadScreenController

screens = {
    "capture screen": {
        "controller": CaptureScreenController,
    },

    "calibrate screen": {
        "controller": CalibrateScreenController,
    },

    "distance screen": {
        "controller": DistanceScreenController,
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
