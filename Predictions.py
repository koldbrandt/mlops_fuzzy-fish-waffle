import io
import httpx
from PIL import Image
import src.data.get_dataset
from omegaconf import OmegaConf
import os
import cv2
import numpy as np

def main():

    img_file = open("./im00.png", "rb").read()
    # img_file = Image.open(r"./im00.png") 
    # print(img)
    # image = np.expand_dims(img, axis=0)
    # print(image.shape)
    # im = np.swapaxes(img, 0, 2)
    # print(img_file)
    # print(im.shape)
    # print(np.swapaxes(im, 0, 2).shape)

    res = httpx.post("http://127.0.0.1:8080/predictions/my_fancy_model", data=img_file)
    res.json()
    print(res)
    print(res.json())


if __name__ == "__main__":
        main()

# img_file = open("./im00.png", "rb").read()
# image = Image.open(io.BytesIO(img_file))
# res = httpx.post("http://127.0.0.1:8080/predictions/my_fancy_model", data=images)
# res.json()
# print(res)
# print(res.json())