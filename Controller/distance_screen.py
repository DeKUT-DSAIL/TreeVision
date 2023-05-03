import importlib

import View.DistanceScreen.distance_screen

# We have to manually reload the view module in order to apply the
# changes made to the code on a subsequent hot reload.
# If you no longer need a hot reload, you can delete this instruction.
importlib.reload(View.DistanceScreen.distance_screen)




class DistanceScreenController:
    """
    The `DistanceScreenController` class represents a controller implementation.
    Coordinates work of the view with the model.
    The controller implements the strategy pattern. The controller connects to
    the view to control its actions.
    """

    def __init__(self):
        self.view = View.DistanceScreen.distance_screen.DistanceScreenView(controller=self)

    def get_view(self) -> View.DistanceScreen.distance_screen:
        return self.view
