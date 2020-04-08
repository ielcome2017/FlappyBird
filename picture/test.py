import pandas as pd

data = [
    ["input_state", "", "", "", "(None, 80, 80, 4)"],
    ["input_action", "", "", "", "(None, 2)"],
    ["卷积(conv)", "(None, 80, 80, 4)", "(8, 8, 4, 32)", "4", "(None, 20, 20, 32)"],
    ["池化(pool)", "(None, 20, 20, 32)", "", "2", "(None, 10, 10, 32)"],
["卷积(conv)", "(None, 20, 20, 32)", "(4, 4, 32, 64)", "2", "(None, 5, 5, 64)"],
["卷积(conv)", "(None, 5, 5, 64)", "(3, 3, 64, 64)", "1", "(None, 5, 5, 64)"],
    ["flatten", "", "", "", "(None, 1600)"],
    ["全连接(fully_connect)", "1600", "(1600, 512)", "", "(None, 512)"],
    ["全连接(fully_connect)", "512", "(512, 2)", "", "(None, 2)"],
    ["点乘(dot)", "(None, 2)", "", "", "(None, 1)"]
]

df = pd.DataFrame(data, columns=["Layer (type)", "Input Shape", "Kernel", "stride", "Output Shape"])
df.to_html("picture/param.html")