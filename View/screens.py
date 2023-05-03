# The screens dictionary contains the objects of the models and controllers
# of the screens of the application.


from Controller.capture_screen import CaptureScreenController
from Controller.calibrate_screen import CalibrateScreenController
from Controller.distance_screen import DistanceScreenController
from Controller.extract_screen import ExtractScreenController

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
}