import pygame
import numpy as np



def check_crash(player, element):
    area = player.mask.overlap(element.mask, (int(element.x - player.x), int(element.y-player.y)))
    if area is not None:
        return True
    return False


class Element:
    def __init__(self, filename, x=0, y=0, rotate=False):
        self.image = pygame.image.load(filename)
        if rotate:
            self.image = pygame.transform.flip(self.image, True, True)
        self.x = x
        self.y = y
        self.mask = pygame.mask.from_surface(self.image)

        self.info = (self.x, self.y)

    def move(self, offset_x, offset_y):
        self.x += offset_x
        self.y += offset_y
        self.info = (self.x, self.y)

    def update(self, player):
        detach = False
        flag = check_crash(player, self)
        return np.array([flag, detach, False])


class PlayElement:
    def __init__(self, x, y, bottom):
        self.data = self.generator() # 生成图像
        self.image, self.mask = next(self.data)

        self.x, self.y = x, y
        self.max_y = bottom - self.image.get_height()    # 到达底端
        self.vely, self.accy = 0, 1 # 初始速度和增速

        self.max_vely, self.min_vely = 10, -8   # 向上向下的速度极限
        self.flap_acc = -9  # 按键后将往上飞

        self.info = (x, y)

        self.max_observe = 4
        self.observes = []

    def generator(self):
        states = [
            pygame.image.load("assets/sprites/redbird-upflap.png"),
            pygame.image.load("assets/sprites/redbird-midflap.png"),
            pygame.image.load("assets/sprites/redbird-downflap.png")
        ]
        mask = [pygame.mask.from_surface(s) for s in states]
        while True:
            for i in [0, 1, 2, 1]:
                yield states[i], mask[i]

    def update(self, actors):
        self.x += 0
        self.y = self.y + self.vely     # min(self.y + self.vely, self.max_y)

        self.vely = self.vely + self.accy if actors[1] == 0 else self.flap_acc  # 更新速度
        self.image, self.mask = next(self.data)     # 获取当前图形和mask
        self.info = (self.x, self.y)


class PipeElement:
    def __init__(self, x, offset):
        self.down = Element("assets/sprites/pipe-green.png", x)
        offset_y = 0.2*offset - np.random.randint(2, 10) * 10 - self.down.image.get_height()
        self.down.move(0, -offset_y)

        self.up = Element("assets/sprites/pipe-green.png", x, rotate=True)
        offset_y = self.down.y - 100 - self.up.image.get_height()
        self.up.move(0, offset_y)

        self.velx = -4

    def update(self, player):
        self.up.move(self.velx, 0)    # 移动
        self.down.move(self.velx, 0)
        detach = 0 < self.up.x < 5  # 是否到达左边界
        flag = check_crash(player, self.down) or check_crash(player, self.up)
        player_mid_pos = player.x + player.image.get_width()/2  # 是否得分
        pipe_mid_pos = self.up.x + self.up.image.get_width()/2

        score = False
        if pipe_mid_pos <= player_mid_pos < pipe_mid_pos + 4:
            score = True

        return np.array([flag, detach, score])


class ScoreElement:
    def __init__(self, x, y):
        self.images = []
        self.x, self.y = x-30, y + 30
        self.score = 0

    def update(self, values):

        self.score += values
        self.images = []

        values = str(self.score)

        for i, v in enumerate(values):
            element = Element("assets/sprites/{}.png".format(v), self.x, self.y)
            element.move(i*22, 0)
            self.images.append(element)


