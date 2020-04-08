import numpy as np
# 设置引擎后端 pyqt5
backen = "pygame"

if backen == "pygame":
    import pygame

    def flip(image, xbool=True, ybool=True):
        return pygame.transform.flip(image, xbool, ybool)

    def key_event(event=pygame.event):
        for e in event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                break
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_UP:
                    return True
        return False

    class Display:
        def __init__(self):
            pygame.init()
            self.display = pygame.display
            self.clock = pygame.time.Clock()
            self.set_mode = pygame.display.set_mode
            self.set_caption = pygame.display.set_caption
            self.update = pygame.display.update
            self.get_surface = pygame.display.get_surface

        def get_event(self, ind):
            """
            :param ind: 为电脑默认或者AI操作，但是键盘案件操作优先，因此以下响应时该项会被覆盖
            :return: 返回动作索引，若出现键盘事件则覆盖之前
            """
            action = np.zeros(2)
            flag = key_event()
            if flag:
                ind = 1
            action[ind] = 1
            return action

    load_image = lambda x: pygame.image.load(x).convert_alpha()
    convert_data = pygame.surfarray.array3d
    get_mask = pygame.mask.from_surface
    GameOver = pygame.error

else:
    from game.display import load_image, flip, get_mask, convert_data
    from game.display import Display, Clock, GameOver

