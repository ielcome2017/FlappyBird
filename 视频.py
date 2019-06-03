from PyQt5 import QtCore 
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QPainter
import os

class Player(QWidget):
    def __init__(self, parent=None, flags=Qt.WindowFlags()):
        super(Player, self).__init__()
        self.resize(500, 500)
        pth = "assets/sprites"
        self._files = [os.path.join(pth, file) for file in os.listdir(pth)]

        self._pixmaps = [QPixmap(file) for file in self._files]
        self._index = 0

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(300)
    
    def paintEvent(self, e):
        print(self._index)
        p = QPainter()
        p.begin(self)
        pixmap = self._pixmaps[self._index]
        self._index = (self._index + 1) % len(self._pixmaps)
        p.drawPixmap(self.rect(), pixmap)
        p.end()

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    form = Player()
    form.show()
    sys.exit(app.exec_())