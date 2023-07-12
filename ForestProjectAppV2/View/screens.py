from Controller.image_screen import ImageScreenController
from Controller.main_screen import MainScreenController
from Controller.upload_screen import UploadScreenController

screens = {
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
