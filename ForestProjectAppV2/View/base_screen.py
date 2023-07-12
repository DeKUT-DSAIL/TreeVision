from kivy.properties import ObjectProperty

from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen


class BaseScreenView(MDScreen):
    controller = ObjectProperty()
    model = ObjectProperty()
    manager_screens = ObjectProperty()

    def __init__(self, **kw):
        super().__init__(**kw)
        self.app = MDApp.get_running_app()
        self.manager_screens = self.app.manager_screens
        # Adding a view class as observer.
        # self.model.add_observer(self)
