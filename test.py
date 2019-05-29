# coding: utf-8

# In[ ]:
import pygame
import os

# In[]
class Resource(object):
    class Hitmask(object):
        def __init__(self, pipe, player):
            self.pipe = (self.git_hitmask(pipe[0]), self.git_hitmask(pipe[1]))
            self.player = (
                self.git_hitmask(player[0]),
                self.git_hitmask(player[1]),
                self.git_hitmask(player[2])
            )
        
        def git_hitmask(self, image):
            mask = []
            for x in range(image.get_width()):
                mask.append([])
                for y in range(image.get_height()):
                    mask[x].append(bool(image.get_at((x, y))[3]))
            return mask

    def __init__(self, pth="assets/sprites"):
        
        self._numbers = [os.path.join(pth, file) for file in os.listdir(pth) 
                    if str.isdigit(os.path.splitext(file)[0])]
        self._players = [os.path.join(pth, file) for file in os.listdir(pth)
                        if str.startswith(file, "redbird")]
        self._background = os.path.join(pth, "background-black.png")
        self._pipe = os.path.join(pth, "pipe-green.png")
        self._base = os.path.join(pth, "base.png")

    def numbers(self):
        return [pygame.image.load(number).convert_alpha()
                        for number in self._numbers]
    
    def players(self):
        return [pygame.image.load(player).convert_alpha()
                        for player in self._players]
    def base(self):
        return pygame.image.load(self._base).convert_alpha()
    
    def pipe(self):
        return (
            pygame.transform.rotate(pygame.image.load(self._pipe).convert_alpha(), 180),
            pygame.image.load(self._pipe).convert_alpha()
        )
    
    def background(self):
        return pygame.image.load(self._background).convert_alpha()
    
    def hitmask(self):
        return self.Hitmask(self.pipe(), self.players())

    


# In[ ]:


from itertools import cycle
class Player:
    def __init__(self, image, x, y):
        self.image = image
        self.width = image[0].get_width()
        self.height = image[0].get_height()

        # set Pos
        self.x = x 
        self.y = y
        self.index = 0
        self.player_index_gen = cycle([0, 1, 2, 1])
        self.vely = 0
        self.max_vely = 10
        self.min_vely = -8
        self.accy = 1
        self.flap_acc = -9
        
    
    def at(self, index):
        return self.image[index]

class Pipe:
    def __init__(self, image, x, y):
        self.image = image
        self.width = image.get_width()
        self.height = image.get_height()
        
        self.x = x
        self.y = y
        self.velx = -4

class Base:
    def __init__(self, image, x, y):
        self.image = image
        self.width = image.get_width()
        self.height = image.get_height()
        self.x = x
        self.y = y
        self.shift = 0

    def set_shift(self, width):
        self.shift = self.width - width

class Background:
    def __init__(self, image, x=0, y=0):
        self.image = image
        self.width = image.get_width()
        self.height = image.get_height()
        self.x = x
        self.y = y

# In[ ]:
import random
pygame.init()
class GameState:
    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y
    res = Resource()
    def __init__(self):
        self.fps = 30
        self.fps_clock = pygame.time.Clock()
        # screen 
        self.width = 288
        self.height = 512
        self.screen = pygame.display.set_mode((self.width, self.height))
        
        self.pipe_gap_size = 100
        
        self.score = self.loop_iter = 0

        res = self.res
        self.background = Background(res.background())
        self.player = Player(res.players(), int(self.width*0.2), int((self.height - res.players()[0].get_height()) / 2))
        self.base = Base(res.base(), 0, self.height*0.79)
        self.base.set_shift(self.background.width)
        self.hitmask = res.hitmask()

        new_pipe1 = self.get_random_pipe(self.width)
        new_pipe2 = self.get_random_pipe(self.width+ (self.width//2))

        self.up_pipes = [new_pipe1[0], new_pipe2[0]]
        self.low_pipes = [new_pipe1[1], new_pipe2[1]]

        self.player_falpped = False
    
    def get_random_pipe(self, x):
        ys = [i*10 for i in range(2, 10)]
        idx = random.randint(0, len(ys)-1)
        y = ys[idx]

        y += int(self.base.y * 0.2)

        return Pipe(self.res.pipe()[0], x, y-self.res.pipe()[0].get_height()), Pipe(self.res.pipe()[1], x, y+self.pipe_gap_size)

    def frame_step(self, actions):
        pygame.event.pump()
        reward = 0.1
        terminal = False
        if actions[1] == 1:
            if self.player.y > -2 * self.player.height:
                self.player.vely = self.player.flap_acc
                self.player_falpped = True
        
        player_mid_pos = self.player.x + self.player.width / 2
        for pipe in self.up_pipes:
            pipe_mid_pos = pipe.x + pipe.width / 2
            if pipe_mid_pos <= player_mid_pos < pipe_mid_pos + 4:
                self.score += 1
                reward = 1
        
        if (self.loop_iter + 1) % 3 == 0:
            self.player.index = next(self.player.player_index_gen)
        
        self.loop_iter = (self.loop_iter + 1) % 30
        self.base.x = -((-self.base.x + 100) % self.base.shift)

        if self.player.vely < self.player.max_vely and not self.player_falpped:
            self.player.vely += self.player.accy
        if self.player_falpped:
            self.player_falpped = False
        self.player.y += min(self.player.vely, self.base.y - self.player.y - self.player.height)
        self.player.y = max(0, self.player.y)
        for u, l in zip(self.up_pipes, self.low_pipes):
            u.x += u.velx
            l.x += u.velx
        if 0 < self.up_pipes[0].x < 5:
            new_pipe = self.get_random_pipe(self.width+10)
            self.up_pipes.append(new_pipe[0])
            self.low_pipes.append(new_pipe[1])
        if self.up_pipes[0].x < - self.up_pipes[0].width:
            self.up_pipes.pop(0)
            self.low_pipes.pop(0)
        
        is_crash = self.check_crash(self.player, self.up_pipes, self.low_pipes)
        if is_crash:
            terminal = True
            self.__init__()
            reward = -1
        self.screen.blit(self.background.image, (self.background.x, self.background.y))
        for u, l in zip(self.up_pipes, self.low_pipes):
            self.screen.blit(u.image, (u.x, u.y))
            self.screen.blit(l.image, (l.x, l.y))
        
        self.screen.blit(self.base.image, (self.base.x, self.base.y))

        self.screen.blit(self.player.at(self.player.index), (self.player.x, self.player.y))

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        self.fps_clock.tick(self.fps)
        return image_data, reward, terminal


    def check_crash(self, player: Player, up_pipes, low_pipes):
        pi = player.index
        if player.y + player.width > self.base.y - 1:
            return True
        player_rect = pygame.Rect(player.x, player.y, player.width, player.height)
        for u, l in zip(up_pipes, low_pipes):
            urect = pygame.Rect(u.x, u.y, u.width, u.height)
            lrect = pygame.Rect(l.x, l.y, l.width, l.height)

            phitmask = self.hitmask.player[pi]
            uhitmask = self.hitmask.pipe[0]
            lhitmask = self.hitmask.pipe[1]

            ucollide = self.pixel_collision(player_rect, urect, phitmask, uhitmask)
            lcollide = self.pixel_collision(player_rect, lrect, phitmask, lhitmask)

            if ucollide or lcollide:
                return True

        return False
    
    def pixel_collision(self, rect1, rect2, hitmask1, hitmask2):
        rect = rect1.clip(rect2)

        if rect.width == 0 or rect.height == 0:
            return False

        x1, y1 = rect.x - rect1.x, rect.y - rect1.y
        x2, y2 = rect.x - rect2.x, rect.y - rect2.y

        for x in range(rect.width):
            for y in range(rect.height):
                if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                    return True
        return False




# In[ ]:
import numpy as np 
import threading
game = GameState()

while 1:
    b = [1, 0]
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            pygame.quit()
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_UP:
                b = np.array([0, 1])
                
    game.frame_step(b)



#%%
