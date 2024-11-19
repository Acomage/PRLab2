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
