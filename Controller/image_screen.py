import importlib

import View.ImageScreen.image_screen

# We have to manually reload the view module in order to apply the
# changes made to the code on a subsequent hot reload.
# If you no longer need a hot reload, you can delete this instruction.
importlib.reload(View.ImageScreen.image_screen)


class ImageScreenController:
    """
    The `MainScreenController` class represents a controller implementation.
    Coordinates work of the view with the model.
    The controller implements the strategy pattern. The controller connects to
    the view to control its actions.
    """

    speed_dial_data = {
        'Stereographic': [
            'language-php',
            "on_press", lambda x: print("pressed PHP"),
        ],
        'Mask': [
            'palm-tree',
            "on_press", lambda x: print("pressed"),
        ],
    }

    def __init__(self):
        self.view = View.ImageScreen.image_screen.ImageScreenView(controller=self)

    def change_default_tab(self, *args):
        self.view.ids.bottom_navigation.switch_tab("screen 2")

    def switch_screen(self, screen):
        if isinstance(screen, str):
            self.view.manager_screens.current = screen
        else:
            self.view.manager_screens.switch_to(screen)

    def next_image(self, instance, image_path):
        if "RIGHT" in image_path:
            # Replace right with left
            image_path = image_path.replace("RIGHT", "LEFT")

        elif "LEFT" in image_path:
            # Replace left with right
            image_path = image_path.replace("LEFT", "RIGHT")

        image_name = image_path.split("/")[-1]

        instance.image_source = image_path
        instance.image_name = image_name

    def start_animation(self, args):
        print(args)
        print(self.view.ids)

    def get_view(self) -> View.ImageScreen.image_screen:
        return self.view
