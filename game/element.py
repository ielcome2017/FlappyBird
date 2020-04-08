import numpy as np
from game.engine import load_image, flip, get_mask


def check_crash(player, element):
    area = player.mask.overlap(element.mask, (int(element.x - player.x), int(element.y-player.y)))
    if area is not None:
        return True
    return False


class Element:
    def __init__(self, filename, x=0, y=0, rotate=False, other=0):
        self.image = load_image(filename)
        if rotate:
            self.image = flip(self.image)
        self.x = x
        self.y = y
        self.mask = get_mask(self.image)

        self.info = (self.x, self.y)
        self.other = other

    def move(self, offset_x, offset_y):
        self.x += offset_x
        self.y += offset_y
        self.info = (self.x, self.y)

    def update(self, player):
        detach = False
        flag = check_crash(player, self)
        return np.array([flag, detach, False])


class BaseElement(Element):

    def update(self, player):
        shift = self.image.get_width() - self.other
        self.x = -((-self.x + 100) % shift)
        self.info = [self.x, self.y]
        return super(BaseElement, self).update(player)


class PlayElement:
    def __init__(self, x, y, bottom):
        # 加载资源
        self.images = [
            load_image("assets/sprites/redbird-upflap.png"),
            load_image("assets/sprites/redbird-midflap.png"),
            load_image("assets/sprites/redbird-downflap.png")
        ]
        self.masks = [get_mask(s) for s in self.images]
        self.data = self.generator()    # 生成图像
        self.image, self.mask = next(self.data)

        self.bottom = bottom
        self.x, self.y = x, int(y - self.image.get_height() / 2)
        self.max_y = bottom - self.image.get_height()    # 到达底端
        self.speed, self.accelerated_speed = 0, 1     # 初始速度和增速
        self.max_speed, self.min_speed = 10, -8       # 向上向下的速度极限
        self.flap_acc = -9      # 按键后将往上飞
        self.info = (x, y)

    def generator(self):
        while True:
            for i in np.array([0, 1, 2, 1]).repeat(3):
                yield self.images[i], self.masks[i]

    def update(self, actors):
        self.speed = self.speed + self.accelerated_speed if actors[1] == 0 else self.flap_acc    # 更新速度
        self.speed = np.clip(self.speed, a_min=self.min_speed, a_max=self.max_speed)

        self.y += self.speed
        self.image, self.mask = next(self.data)     # 获取当前图形和mask
        self.info = (self.x, self.y)


class PipeElement:
    def __init__(self, x, offset):
        self.down = Element("assets/sprites/pipe-green.png", x)
        offset_y = int(0.2*offset) + np.random.randint(2, 10) * 10 + 100
        self.down.move(0, offset_y)

        self.up = Element("assets/sprites/pipe-green.png", x, rotate=True)
        offset_y = offset_y - self.up.image.get_height() - 100
        self.up.move(0, offset_y)

        self.speed = -4

    def update(self, player):
        self.up.move(self.speed, 0)    # 移动
        self.down.move(self.speed, 0)
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
        self.x, self.y = x, y + 40
        self.score = 0

    def update(self, values):
        self.score += values
        self.images = []
        width = []
        values = str(self.score)
        for v in values:
            element = Element("assets/sprites/{}.png".format(v), self.x, self.y)
            width.append(element.image.get_width())
            self.images.append(element)
        start = self.x - (np.sum(width) + 2 * len(width)) / 2
        for num, element in enumerate(self.images):
            element.move(start - element.x, 0)

            start += width[num] + 2



