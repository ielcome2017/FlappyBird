import pygame
from game.element import Element, PlayElement, PipeElement, ScoreElement
from game.control import Control
import numpy as np


class Game:
    def __init__(self):
        pygame.init()
        self.fps = 30   # 定义游戏的fps
        self.clock = pygame.time.Clock()    #游戏时钟

        self.width, self.height = 288, 512
        self.screen = pygame.display.set_mode((self.width, self.height))

        self.background, self.base, self.player, self.scoreboard = self.create()
        self.control, self.state = self.register()

    def create(self):
        background = Element("assets/sprites/background-black.png")
        base = Element("assets/sprites/base.png", 0, self.height * 0.79)
        player = PlayElement(int(self.width*0.2), int(self.height // 2), base.y)
        scoreboard = ScoreElement(self.width // 2, base.y)
        return background, base, player, scoreboard

    def register(self):
        # 观察者注册
        control = Control(self.player, self.scoreboard)
        control.attach(self.base)
        pipe = PipeElement(self.width, offset=self.base.y)
        control.attach(pipe)
        pipe = PipeElement(self.width * 1.5, offset=self.base.y)
        control.attach(pipe)
        # 状态，控制是否结束游戏，是否有新元素加入，分数
        state = np.array([False, False, False])
        return control, state

    def restart(self):
        self.background, self.base, self.player, self.scoreboard = self.create()
        self.control, self.state = self.register()
        return -1

    def frame_step(self, actions):
        self.screen.blit(self.background.image, self.background.info)
        self.screen.blit(self.player.image, self.player.info)

        self.state[0] = self.player.y < 0    # 边缘检测

        state = self.control.notify(actions)    # flag, detach, score

        self.state = np.logical_or(self.state, state)
        if state[1]:
            pipe = PipeElement(self.width + 10, offset=self.base.y)
            self.control.attach(pipe)

        for i in range(1, len(self.control.observes)):
            p = self.control.observes[i]
            self.screen.blit(p.down.image, p.down.info)
            self.screen.blit(p.up.image, p.up.info)

        self.screen.blit(self.base.image, self.base.info)
        for board in self.scoreboard.images:
            self.screen.blit(board.image, board.info)
        pygame.display.update()
        self.clock.tick(self.fps)

        # 游戏输出状态
        reward = self.restart() if self.state[0] else 0.1
        terminal = reward < 0
        reward = max(reward, int(state[-1])) if not terminal else reward
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        return image_data, reward, int(terminal)





