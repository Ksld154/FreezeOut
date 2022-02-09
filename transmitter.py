from threading import Thread
import threading
import time

# TRANSMIT_TIME = 40


class Transmitter(Thread):
    def __init__(self, transmit_time=20) -> None:
        threading.Thread.__init__(self)
        self.transmit_time = transmit_time

    def transmit(self):
        time.sleep(self.transmit_time)

    def run(self) -> None:
        self.transmit()
        # return super().run()
