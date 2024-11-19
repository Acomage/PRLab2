"""
这个文件用于将pdf文件转换为png文件，因为graph_tool可视化时如果不设置成输出pdf，绘图时效果很差，所以先生成pdf再转换为png
"""

from config import image_output_path
import os

for root, dirs, files in os.walk(image_output_path):
    for file in files:
        if file.endswith(".pdf"):
            print(os.path.join(root, file))
            file_name = file.split(".")[0]
            os.system(
                f"magick -density 300 {os.path.join(root, file)} -quality 100 {os.path.join(root, file_name)}.png"
            )
