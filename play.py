import pygame
from game.element import Element, PlayElement, PipeElement, ScoreElement, BaseElement
from game.control import Control
import numpy as np


def get_event(ind):
    """
    :param ind: 为电脑默认或者AI操作，但是键盘案件操作优先，因此以下响应时该项会被覆盖
    :return: 返回动作索引，若出现键盘事件则覆盖之前
    """
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            pygame.quit()
            break
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_UP:
                ind = 1
    return ind


class Game:
    def __init__(self):
        pygame.init()
        self.width, self.height = 288, 512  # 定义游戏界面大小
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Flappy Bird')
        self.fps = 30   # 定义游戏的fps
        self.clock = pygame.time.Clock()   # 游戏时钟

        self.background, self.base, self.player, self.scoreboard = self.create()
        self.control, self.state = self.register()
        self.get_event = get_event

    def create(self):
        """
        :return: 创建游戏各元素
        """
        background = Element("assets/sprites/background-black.png")
        base = BaseElement("assets/sprites/base.png", 0, self.height * 0.79, False, background.image.get_width())
        player = PlayElement(int(self.width*0.2), int(self.height / 2), base.y)
        scoreboard = ScoreElement(self.width // 2, base.y)
        return background, base, player, scoreboard

    def register(self):
        """
        游戏各元素如何更新依赖Control类控制，游戏开始加入BASE, 两个PIPE，Player和计分板，后两个较特殊在构造时增加
        state [False, False, False] 表示游戏没有结束，无新管道加入，分数没有增加
        :return:
        """
        control = Control(self.player, self.scoreboard)
        control.attach(self.base)
        pipe = PipeElement(self.width, offset=self.base.y)
        control.attach(pipe)
        pipe = PipeElement(self.width * 1.5, offset=self.base.y)
        control.attach(pipe)
        state = np.array([False, False, False])     # 状态，控制是否结束游戏，是否有新元素加入，分数
        return control, state

    def restart(self):
        self.background, self.base, self.player, self.scoreboard = self.create()
        self.control, self.state = self.register()
        return -1

    def draw_element(self, element):
        self.screen.blit(element.image, element.info)

    def draw(self):
        elements = [self.background, self.player]
        for p in self.control.observes[1:]:
            elements.append(p.up)
            elements.append(p.down)
        elements.extend([self.base, *self.scoreboard.images])
        for elem in elements:
            self.draw_element(elem)

    def frame_step(self, actions):
        pygame.event.pump()

        self.state[0] = self.player.y < 0    # 小鸟到达边缘游戏借宿，state第一个标志位为游戏结束标志
        state = self.control.notify(actions)    # 根据动作更新state
        self.state = np.logical_or(self.state, state)
        if state[1]:    # state第二个标志位表示是否刷新新的管道
            pipe = PipeElement(self.width + 10, offset=self.base.y)
            self.control.attach(pipe)
        reward = self.restart() if self.state[0] else 0.1   # 游戏输出状态，没有分数增加或者游戏未借宿则未0.1
        terminal = reward < 0
        reward = max(reward, int(state[-1])) if not terminal else reward    # state第三个标志位表示分数是否增加

        self.draw()     # 绘制游戏界面
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        self.clock.tick(self.fps)
        return image_data, reward, int(terminal)


if __name__ == '__main__':
    game = Game()
    while True:
        action_ind = 0
        action_ind = game.get_event(action_ind)
        action = np.zeros(2)
        action[action_ind] = 1
        game.frame_step(action)



