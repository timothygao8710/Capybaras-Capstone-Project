import cv2
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('bmh')

import os
import math

#image_path: [string] path to the image
#with_plot: [bool] whether to plot stuff
#grayScale: [bool] whether the image is gray scale

def enhance_contrast(image_path, with_plot=True, gray_scale=False):
    def read_this(image_file, gray_scale=False):
        image_src = cv2.imread(image_file)
        if gray_scale:
            image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
        else:
            image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
        return image_src
    # equalizing image
    def equalize_this(image_file, with_plot=False, gray_scale=False):
        image_src = read_this(image_file=image_file, gray_scale=gray_scale)
        if not gray_scale:
            r_image, g_image, b_image = cv2.split(image_src)

            r_image_eq = cv2.equalizeHist(r_image)
            g_image_eq = cv2.equalizeHist(g_image)
            b_image_eq = cv2.equalizeHist(b_image)

            image_eq = cv2.merge((r_image_eq, g_image_eq, b_image_eq))
            cmap_val = None
        else:
            image_eq = cv2.equalizeHist(image_src)
            cmap_val = 'gray'

        if with_plot:
            fig = plt.figure(figsize=(10, 20))

            ax1 = fig.add_subplot(2, 2, 1)
            ax1.axis("off")
            ax1.title.set_text('Original')
            ax2 = fig.add_subplot(2, 2, 2)
            ax2.axis("off")
            ax2.title.set_text("Equalized")

            ax1.imshow(image_src, cmap=cmap_val)
            ax2.imshow(image_eq, cmap=cmap_val)
            #return True
        return image_eq

    def plot_image(img, title=""):
        plt.title(title)
        
        if img.shape[0] <= 32:
            plt.xticks(range(img.shape[0]))
            plt.yticks(range(img.shape[1]))
        
        plt.imshow(img, extent=[0, img.shape[0], img.shape[1], 0], cmap='viridis')
        plt.show()

    img_eq = equalize_this(image_file=image_path,with_plot=False,gray_scale=False)

    # grayscaling the image
    gray_img=cv2.cvtColor(img_eq,cv2.COLOR_BGR2GRAY)

    # calculating histograms
    hist=cv2.calcHist(gray_img,[0],None,[256],[0,256])

    # equalizing histograms
    gray_img_eqhist=cv2.equalizeHist(gray_img)
    hist=cv2.calcHist(gray_img_eqhist,[0],None,[256],[0,256])

    # thresholding
    th=80
    max_val=255

    # ret, o1 = cv2.threshold(gray_img_eqhist, th, max_val, cv2.THRESH_BINARY)
    ret, o2 = cv2.threshold(gray_img_eqhist, th, max_val, cv2.THRESH_BINARY_INV)
    # ret, o3 = cv2.threshold(gray_img_eqhist, th, max_val, cv2.THRESH_TOZERO)
    # ret, o4 = cv2.threshold(gray_img_eqhist, th, max_val, cv2.THRESH_TOZERO_INV)
    # ret, o5 = cv2.threshold(gray_img_eqhist, th, max_val, cv2.THRESH_TRUNC)
    # ret ,o6=  cv2.threshold(gray_img_eqhist, th, max_val,  cv2.THRESH_OTSU)
    plot_image(o2, "enhanced image")
    return o2
    
if __name__ == "__main__":
    # img = Image.open(os.path.join("test_grey.jpg"))
    # enhance_contrast(os.path.join("test_images", "squarecapybara.jpg"))
    enhance_contrast(os.path.join("test_images", "timothycapybara.png"))