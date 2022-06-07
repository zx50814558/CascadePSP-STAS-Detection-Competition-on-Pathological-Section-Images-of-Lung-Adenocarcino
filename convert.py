import os
import pandas as pd
import numpy as np
import shutil

# 轉換 image 格式
data_input = 'input'
os.makedirs(data_input, exist_ok=True)
# 轉換 Mask 格式
for i in os.listdir(data_input):
    if i[-3:] == "png":
        new_name = i[:-4] + "_seg" + ".png"
        shutil.move(os.path.join(data_input, i), os.path.join(data_input, new_name))

for i in os.listdir(data_input):
    if i[-3:] == "jpg":
        new_name = i[:-4] + "_im" + ".jpg"
        shutil.move(os.path.join(data_input, i), os.path.join(data_input, new_name))


