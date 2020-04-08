import sys
import time
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtCore import QRect
from PyQt5.QtGui import QPainter, QPixmap
from PyQt5.QtWidgets import QWidget, QApplication
import threading
from PIL import Image
from queue import Queue

MAX_LENGTH = 1000


def convert_data(x):
    return np.transpose(x, [1, 0, 2])


def get_mask(image):
    return _Mask(np.array(image.data)[:, :, -1])


def load_image(filename):
    im = Image.open(filename)
    return _Image(im)


def flip(image):
    return image.rotate(180)


class _Image:
    def __init__(self, data, size="size"):
        self.data = data
        self.width, self.height = getattr(data, size)

    def rotate(self, num):
        self.data = self.data.rotate(num)
        return self

    def get_height(self):
        return self.height

    def get_width(self):
        return self.width


class _Mask(_Image):
    def __init__(self, data):
        super().__init__(data, size="shape")
        self.width, self.height = self.height, self.width

    def overlap(self, other, xy):
        x0, y0, x1, y1 = 0, 0, xy[0], xy[1]
        x, y = max(x0, x1), max(y0, y1)
        xx = min(x0 + self.get_width(), x1 + other.get_width())
        yy = min(y0 + self.get_height(), y1 + other.get_height())
        rect = (x, y, max(xx - x, 0), max(yy - y, 0))
        if rect[2] == 0 or rect[3] == 0:
            return None
        x1, y1 = rect[0] - x0, rect[1] - rect[1]
        if np.count_nonzero(self.data[x1: x1 + rect[2], y1: y1 + rect[3]]) or \
                np.count_nonzero(other.data[x1: x1 + rect[2], y1: y1 + rect[3]]):
            return np.argwhere(self.data[x1: x1 + rect[2], y1: y1 + rect[3]] > 0)
        return None


class Form(QWidget):
    def __init__(self, parent=None, size=(288, 512), title="Python", frames=None, clock=None, response=None,
                 surface=None):
        super(Form, self).__init__(parent)
        self.resize(*size)
        self.setWindowTitle(title)
        self.frames = frames
        self.clock = clock
        self.response = response
        self.surface = surface
        self._state = False
        self.picture = QPixmap(self.size())
        th = threading.Thread(target=self.paint)
        th.start()

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.response.put("QUIT")
        self.surface.put(np.array(Image.fromqpixmap(self.picture)))
        self._state = True

    def paint(self):
        while True:
            if self._state:
                break
            sec = self.clock.get()
            image = self.frames.get()
            self.draw(image)
            self.surface.put(np.array(Image.fromqpixmap(self.picture)))
            self.update()
            time.sleep(1 / sec)

    def draw(self, image):
        try:
            painter = QPainter()
            painter.begin(self.picture)
            for im, [x, y] in image:
                im = im.data.toqpixmap()
                rect = QRect(x, y, im.width(), im.height())
                painter.drawPixmap(rect, im)
            painter.end()
        except Exception as e:
            print(e)

    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        try:
            painter = QPainter()
            painter.begin(self)
            painter.drawPixmap(self.rect(), self.picture)
            painter.end()
        except Exception as e:
            print(e)

    def keyPressEvent(self, a0: QtGui.QKeyEvent) -> None:
        self.response.put("KEY_UP", block=False)


class Screen:
    def __init__(self):
        self.data = []

    def blit(self, im: _Image, xy):
        self.data.append([im, xy])

    def clear(self):
        self.data = []


class Clock(Queue):
    def __init__(self):
        super(Clock, self).__init__()
        self.state = False

    def tick(self, fps):
        self.put(fps, timeout=1 / fps)


class Display:

    def __init__(self):
        super(Display, self).__init__()
        self.screen = Screen()

        self.clock = Clock()
        self.event = Queue()
        self.surface = Queue()
        self.frames = Queue()

        self.state = False

        self.title = None

    def set_caption(self, title):
        self.title = title

    def set_mode(self, size):
        th = threading.Thread(target=self.run, args=[size,])
        th.start()
        return self.screen

    def run(self, size):
        app = QApplication(sys.argv)
        form = Form(size=size,
                    title=self.title,
                    frames=self.frames,
                    clock=self.clock,
                    response=self.event,
                    surface=self.surface)
        form.show()
        app.exec_()

    def update(self):
        self.frames.put(self.screen.data)
        self.screen.clear()

    def get_surface(self):
        return self.surface.get()

    def get_event(self, ind):
        action = np.zeros(2)
        event = self.event
        if not event.empty():
            e = event.get(timeout=0)
            if e == "QUIT":
                raise GameOver
            if e == "KEY_UP":
                ind = 1
        action[ind] = 1
        return action


class GameOver(Exception):
    def __str__(self):
        return "GameOver"
