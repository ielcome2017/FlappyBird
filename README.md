## 强化学习——FlappyBird

### 文件结构
net.py 网络结构包含两个网络V1和V2, 目前只训练V1 \
main.py 训练文件 \
agent.py 产生数据 \
play.py 游戏运行包 \ 
game\control.py 游戏调度文件 \
game\element.py 游戏配置文件 \
game\engine.py 游戏后端，两种选择PyQt5和pygame \
game\display.py PyQt5写的游戏引擎\

### 网络结构 net.py
pass
### 数据产生 agent.py
产生数据的文件 \
所有的数据均由该文件下GameMemory中的generator函数产生，该函数将产生一个生成器，
 