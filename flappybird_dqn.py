#%%
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication
from PyQt5.QtCore import Qt, QRect, QTimer, QThread, pyqtSignal, QBuffer, QByteArray
from PyQt5.QtGui import QPaintEvent, QPainter, QPixmap, QTransform, QKeyEvent
import random
import threading
import os, time
from itertools import cycle
from PIL import Image
import numpy as np 
import cv2
from collections import deque
#%%
def robust(actual_do):
    def add_robust(*args, **keyargs):
        try:
            return actual_do(*args, **keyargs)
        except Exception as e:
            print ('Error execute: %s' % actual_do.__name__, e)
            #traceback.print_exc()
    return add_robust
#%%
class Game(QtCore.QObject):
    sign_step = pyqtSignal(object)
    sign_score = pyqtSignal(int)
    def __init__(self, parent=None):
        super(Game, self).__init__(parent)
        im0 :Image.Image = Image.open("assets/sprites/pipe-green.png")
        im1 = im0.transpose(Image.ROTATE_180)

        self.lmask = np.array(im0)[:, : 3]
        self.umask = np.array(im1)[:, : 3]
        self.img_lpipe = im0.toqpixmap()
        self.img_upipe = im1.toqpixmap()
  
        self.img_background = QPixmap("assets/sprites/background-black.png")
        self._urects = list()
        self._lrects = list()

        pth = "assets/sprites"
        self._birds = [os.path.join(pth, file) for file in os.listdir(pth)
            if str.startswith(file, "redbird")]
        self.img_birds = list()
        self.bird_acts = cycle([0, 1, 2, 1])
        self.bird = QPixmap()
        self.bird_rect = list()
        
        self.pipe_vx = self.bird_vy = self.bird_vy_delta = self.bird_v = int()
        self.score = int()

        self.start()
    
    def width(self):
        return 288
    def height(self):
        return 428
    def rect(self):
        return QRect(0, 0, self.width(), self.height())
    def start(self):
        self._urects = []
        self._lrects = []
        pip1 = self._random(self.width())
        pip2 = self._random(self.width() + self.width()//2)

        self._urects.append(pip1[1])
        self._urects.append(pip2[1])
        self._lrects.append(pip1[0])
        self._lrects.append(pip2[0])
        act = next(self.bird_acts)
        self.bird.load(self._birds[act])
        self.bird_rect = [
            int(self.width()*0.2), 
            (self.height() + self.bird.height())//2, 
            self.bird.width(),
            self.bird.height()]

        self.pipe_vx = -4
        self.bird_vy = 0
        self.bird_vy_delta = 1
        self.bird_v = -9

        self.score = 0

    
    @robust
    def frame_step(self, action = [0, 0]):
        if action[1] == 1:
            self.bird_vy = self.bird_v
        terminal = False
        reward = 0.1
        # 分数
        bird_mid_pos = self.bird_rect[0] + self.bird_rect[2] /2
        for pip in self._lrects:
            pip_mid_pos = pip[0] + self.img_upipe.width()/2
            if pip_mid_pos <= bird_mid_pos < pip_mid_pos+4:
                self.score += 1
                reward = 1
                self.sign_score.emit(self.score)
        if self.check_crash():
            reward = -1
            terminal = True
            self.start()

        # bird位置更新
        self.bird_vy += self.bird_vy_delta
        self.bird_rect[1] = min(self.bird_rect[1] + self.bird_vy, self.height()-self.bird.height())
        self.bird_rect[1] = max(0, self.bird_rect[1])
        # pipe位置更新
        for u, l in zip(self._urects, self._lrects):
            u[0] += self.pipe_vx
            l[0] += self.pipe_vx
        # pipe地图更新
        self._update_pip_rect()

        im = QPixmap(self.width(), self.height())
        # self.render(im)
        self._draw(im)
        self.sign_step.emit(im)
        return [Image.fromqimage(im), reward, terminal, self.score]

    
    def _update_pip_rect(self):
        if 0< self._urects[0][0] < 5:
            pipe = self._random(self.width())
            self._lrects.append(pipe[0])
            self._urects.append(pipe[1])
        if self._urects[0][0] < - self.img_upipe.width():
            self._lrects.pop(0)
            self._urects.pop(0)
    
    @robust
    def check_crash(self):

        if self.bird_rect[1] == 0 or self.bird_rect[1] + self.bird_rect[3] == self.height():
            return True
        bird_rect = self.bird_rect
        bird_mask = np.array(Image.fromqpixmap(self.bird))[:, :, 3]
        for u, l in zip(self._urects, self._lrects):
            urect = (*u, self.img_upipe.width(), self.img_upipe.height())
            lrect = (*l, self.img_lpipe.width(), self.img_lpipe.height())

            ucollision = self.pixel_collision(bird_rect, urect, bird_mask, self.umask)
            lcoliision = self.pixel_collision(bird_rect, lrect, bird_mask, self.lmask)
            if ucollision or lcoliision:
                return True

    def clip(self, rect1, rect2):
        x0, y0, w0, h0 = rect1
        x1, y1, w1, h1 = rect2
        x = max(x0, x1)
        y = max(y0, y1)
        xx = min(x0+w0, x1+w1)
        yy = min(y0+h0, y1+h1)
        return (x, y, max(xx-x, 0), max(yy-y, 0))
    
    def pixel_collision(self, rect1, rect2, hitmask1, hitmask2):
        rect = self.clip(rect1, rect2)

        if rect[2] == 0 or rect[3] == 0:
            return False
        x1, y1 = rect[0] - rect1[0], rect[1] - rect[1]
        x2, y2 = rect[0] - rect2[0], rect[1] - rect2[1]
        if np.count_nonzero(hitmask1[x1: x1+rect[2], y1: y1+rect[3]]) or \
            np.count_nonzero(hitmask2[x1: x1+rect[2], y1: y1+rect[3]]):
            return True
        return False

    def _random(self, x):
        hs = [i*10 for i in range(2, 10)]
        idx = random.randint(0, len(hs)-1)
        h = hs[idx] + self.height() + 100
        ly = h - self.img_upipe.height()
        uy = -self.img_upipe.height() + ly -100
        return [[x, ly], [x, uy]]
                

    def _draw(self, dev):
        p = QPainter()
        p.begin(dev)
        p.setRenderHint(QPainter.Antialiasing)
        # 绘制背景
        p.drawPixmap(self.rect(), self.img_background)

        # 绘制鸟儿
        act = next(self.bird_acts)
        self.bird.load(self._birds[act])
        p.drawPixmap(*self.bird_rect, self.bird)

        # 绘制下管道
        for rect in self._lrects:
            p.drawPixmap(*rect, self.img_lpipe.width(), self.img_lpipe.height(), self.img_lpipe)

        # 绘制上管道
        for rect in self._urects:
            p.drawPixmap(*rect, self.img_upipe.width(), self.img_upipe.height(), self.img_upipe)
        p.end()        
        return dev

#%%
class GamePainter(QWidget):
    def __init__(self, parent=None, flags=Qt.WindowFlags()):
        super(GamePainter, self).__init__(parent)
        self.deque = []
        self.current_im = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._draw)
        self.timer.start(30)
        # self.timer.moveToThread(th)

    @robust
    def frame_step(self, im):
        self.deque.append(im)

    @robust
    def _draw(self):
        if len(self.deque) == 0:
            return
        self.current_im =self.deque.pop(0)
        self.update()
    @robust
    def paintEvent(self, e: QPaintEvent):
        if self.current_im is None:
            return 
        p = QPainter()
        p.begin(self)
        p.drawPixmap(self.rect(), self.current_im)
        p.end()


#%%
class Main(QWidget):
    # sign_step = pyqtSignal()
    class Scoreboard(QtWidgets.QFrame):
        def __init__(self, parent=None, flags=Qt.WindowFlags()):
            super(Main.Scoreboard, self).__init__(parent)

            self.img_scoreboard = QPixmap()
            self.img_scoreboard.load("assets/sprites/base.png")
            self.resize(self.img_scoreboard.width(), self.img_scoreboard.height()*0.2)

            pth = "assets/sprites/"
            self._numbers = [os.path.join(pth, file) for file in os.listdir(pth) 
                    if str.isdigit(os.path.splitext(file)[0])]

           
            self._value = []
            self.set_value(0)
        
        def set_value(self, score):
            self._value = []

            for i in list(str(score)):
                self._value.append(QPixmap(self._numbers[int(i)]))
            
            self.update()
            
        @robust
        def paintEvent(self, e: QPaintEvent):
            p = QPainter()
            p.begin(self)
            p.drawPixmap(self.rect(), self.img_scoreboard)

            x = (self.width()- len(self._value)*self._value[0].width())//2
            y = (self.height() - self._value[0].height())//2
            gap = 4
            for i, im in enumerate(self._value):
                p.drawPixmap(x, y, im.width(), im.height(), im)
                x += im.width() + gap
            p.end()

    def __init__(self, parent=None, flags=Qt.WindowFlags()):
        super(Main, self).__init__(parent)
        self.setGeometry(500, 100, 288, 512)
        
        self.game = Game()
        
        self.game_painter = GamePainter(self)
        self.game_painter.resize(self.size())
        self.scoreboard = Main.Scoreboard()
        self.game_painter.setContentsMargins(0, 0, 0, 0)
        self.scoreboard.setContentsMargins(0, 0, 0, 0)

        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(0)
        layout.addWidget(self.game_painter, stretch=5)
        layout.addWidget(self.scoreboard, stretch=1)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.game.sign_step.connect(self.game_painter.frame_step)
        self.game.sign_score.connect(self.scoreboard.set_value)

        # self.run()

    def run(self):
        th = threading.Thread(target=self.run_event)
        th.setDaemon(True)
        th.start()
    
    def run_event(self):
        for i in range(1000):
            act = np.random.randint(0, 1, size=[2])
            im, _, _, _ = self.game.frame_step(act)
            print(i)

    @robust
    def keyPressEvent(self, e: QKeyEvent):
        if e.key() == Qt.Key_Up:
            # self.sign_step.emit()
            image, reward, terminal, score = self.game.step([0, 1])
            self.scoreboard.set_value(score)

    def _draw(self, p: QPainter):
        scoreboard_rect = QRect(self.scoreboard.x(), self.scoreboard.y(), self.scoreboard.width(), self.scoreboard.height())
        # p.drawPixmap(game_rect, self.img_scoreboard)
        p.drawPixmap(scoreboard_rect, self.img_scoreboard)

    def _rect(self, wdt: QWidget):
        pos = wdt.pos()
        x, y = pos.x(), pos.y()
        return QRect(x, y, x+wdt.width(), y+wdt.height())

#%%
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    form = Main()
    form.show()

    sys.exit(app.exec_())