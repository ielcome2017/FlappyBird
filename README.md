## 强化学习——FlappyBird

### 文件结构
main.py 训练文件 \
agent.py 产生数据 \
play.py 游戏运行包 \ 
control.py 游戏调度文件 \
element.py 游戏配置文件 \

### agent.py
产生数据的文件 \
所有的数据均由该文件下GameMemory中的generator函数产生，该函数将产生一个生成器，
 