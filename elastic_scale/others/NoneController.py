import time

from config.Config import Config


class NoneController:
    def __init__(self, config: Config):
        self.config = config

    def start(self):
        print('NoneController is running...')
        time.sleep(self.config.duration)
