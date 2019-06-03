# libpng warning: iCCP: known incorrect sRGB profile
# 该问题解决方案

import cv2, os
from PyQt5.QtGui import QImage


pth = "assets/sprites/"
files = [os.path.join(pth, file) for file in os.listdir(pth)]
for file in files:
    # im = cv2.imread(file)
    im = QImage(file)
    im.save(file)
    