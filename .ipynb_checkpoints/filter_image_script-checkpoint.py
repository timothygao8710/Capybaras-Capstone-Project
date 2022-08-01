#raw_image: [PIL.PngImagePlugin.PngImageFile] Image as np array
#name: [string] name of the image which will be used
#filter_name: [string] name of filter ('equalized', 'grayed', 'o1', 'o2', 'o3','o4', 'o5', 'o6', 'thresh1', 'thresh2', 'thresh3', 'thresh4')

def filter_image(img_raw, name, filter_name):   
    #Credits to Noah
    import cv2
    import matplotlib.image as mpimg
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import style
    style.use('bmh')

    import os
    import math

    print("Filter image...")
    
    #Functions
    def plot_image(img, title=""):
        plt.title(title)
        
        if img.shape[0] <= 32:
            plt.xticks(range(img.shape[0]))
            plt.yticks(range(img.shape[1]))
        
        plt.imshow(img, extent=[0, img.shape[0], img.shape[1], 0], cmap='viridis')
        plt.show()

    def read_this(image_file, gray_scale=False):
        image_src = cv2.imread(image_file)
        if gray_scale:
            image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
        else:
            image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
        return image_src
    
    def equalize(image_file, with_plot=False, gray_scale=False):
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
    
    # calculating histograms
    hist=cv2.calcHist(gray_img,[0],None,[256],[0,256])
    plt.subplot(121)
    plt.title("Image1")
    plt.xlabel('bins')
    plt.ylabel("No of pixels")
    plt.plot(hist)
    plt.show()
    
    # equalizing histograms
    gray_img_eqhist=cv2.equalizeHist(gray_img)
    hist=cv2.calcHist(gray_img_eqhist,[0],None,[256],[0,256])
    plt.subplot(121)
    plt.plot(hist)
    plt.show()
    
    
    # thresholding
    th=80
    max_val=255

    ret, o1 = cv2.threshold(gray_img_eqhist, th, max_val, cv2.THRESH_BINARY)
    #cv2.putText(o1,"Thresh_Binary",(40,100),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3,cv2.LINE_AA)
    ret, o2 = cv2.threshold(gray_img_eqhist, th, max_val, cv2.THRESH_BINARY_INV)
    #cv2.putText(o2,"Thresh_Binary_inv",(40,100),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3,cv2.LINE_AA)
    ret, o3 = cv2.threshold(gray_img_eqhist, th, max_val, cv2.THRESH_TOZERO)
    #cv2.putText(o3,"Thresh_Tozero",(40,100),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3,cv2.LINE_AA)
    ret, o4 = cv2.threshold(gray_img_eqhist, th, max_val, cv2.THRESH_TOZERO_INV)
    #cv2.putText(o4,"Thresh_Tozero_inv",(40,100),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3,cv2.LINE_AA)
    ret, o5 = cv2.threshold(gray_img_eqhist, th, max_val, cv2.THRESH_TRUNC)
    #cv2.putText(o5,"Thresh_trunc",(40,100),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3,cv2.LINE_AA)
    ret ,o6=  cv2.threshold(gray_img_eqhist, th, max_val,  cv2.THRESH_OTSU)
    #cv2.putText(o6,"Thresh_OSTU",(40,100),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3,cv2.LINE_AA)

    # adaptive thresholding
    thresh1 = cv2.adaptiveThreshold(gray_img_eqhist, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh2 = cv2.adaptiveThreshold(gray_img_eqhist, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 3)
    thresh3 = cv2.adaptiveThreshold(gray_img_eqhist, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 5)
    thresh4 = cv2.adaptiveThreshold(gray_img_eqhist, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 4)

    res_img = img
    if filter_name == "equalized":
        res_img = equalize(img)
    
    if filter_name == 'grayed':
        res_img=cv2.cvtColor(img_eq,cv2.COLOR_BGR2GRAY)
    if filter_name == 'o1':
        res_img=o1
    if filter_name == 'o2':
        res_img=o2
    if filter_name == 'o3':
        res_img=o3
    if filter_name == 'o4':
        res_img=o4
    if filter_name == 'o5':
        res_img=o5
    if filter_name == 'o6':
        res_img=o6
    if filter_name == 'thresh1':
        res_img=thresh1
    if filter_name == 'thresh2':
        res_img=thresh1
    if filter_name == 'thresh3':
        res_img=thresh1
    if filter_name == 'thresh4':
        res_img=thresh1
        
        
    plot_image(res_img, "filtered image")
    
if __name__ == "__main__":
    img = Image.open(os.path.join("test_grey.jpg"))
#     img = Image.open(os.path.join("test_images", "timothycapybara.png"))

    filter_image(img, "image")