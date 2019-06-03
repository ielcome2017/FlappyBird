import numpy as np 
a = [[1, 2, 3], [1,2, 5], [1, 6, 2]]

c = [[7, 2, 3], [1,2, 5], [1, 6, 2]]


a = np.array(a)
c = np.array(c)
d = c-a
d = (np.abs(d) + d)// 2
print(d)

a = np.zeros(10, dtype="int")
a[1] = 1
print(np.count_nonzero(a))

from PIL import Image
import cv2
from PyQt5.QtGui import QPixmap
# with open("assets/sprites/pipe-green.png", "rb") as f:
#     im = f.read()

im = QPixmap("assets/sprites/pipe-green.png")
print(im)
img : Image.Image= Image.fromqpixmap(im)
print(img)

#%%
# 关于引用
class F:
    def __init__(self, *args, **kwargs):
        self.a = 10

class G:
    def __init__(self, *args, **kwargs):
        self.b = 100

f = F()
g = G()
g.b = f.a
g.b = 1000
print(f.a)


#%%
a = [1, 2, 3]
b = a[0]
a[0] = 1000
print(b)

#%%
a = 0
a = list(str(a))
print(a)

#%%
from collections import deque
a = deque()
print(len(a))

#%%
