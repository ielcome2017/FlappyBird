from flappybird_dqn import Main
from PyQt5.QtWidgets import QApplication
import sys, threading, cv2
from PIL import Image

class Flappybird(object):
    def __init__(self, *args, **kwargs):
        self.app = QApplication(sys.argv)
        self.form = Main()
    
    def run(self):
        self.form.show()
        sys.exit(self.app.exec_())

def run():
    for i in range(1000):
        im, _, _, _ = flappy.form.game.frame_step([0, 1])

flappy = Flappybird()

th = threading.Thread(target=run)
th.setDaemon(True)
th.start()

flappy.run()
# im, _, _, _ = flappy.form.game.frame_step([0, 1])
# print(im)