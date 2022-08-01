import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import os
import math

#raw_image: [PIL.PngImagePlugin.PngImageFile] Image as np array
#name: [string] name of the image which will be used
#grid_size: [int] desired grid size
#show: [Int] number of gridded images to show, -1 to show all images

def gridImage(img_raw, name, grid_size = 8, show = -1):    
    #Imports
    import matplotlib.image as mpimg
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import style
    import os
    import math
    style.use('bmh')

    #Functions
    def plot_image(img, title=""):
        plt.title(title)
        
        if img.shape[0] <= 32:
            plt.xticks(range(img.shape[0]))
            plt.yticks(range(img.shape[1]))
        
        plt.imshow(img, extent=[0, img.shape[0], img.shape[1], 0], cmap='viridis')
        plt.show()

    #Main
    img_raw = np.asarray(img_raw)
    plot_image(img_raw)

    n, m, k = img_raw.shape

    if n % grid_size != 0:
        raise Exception(f"Number of rows {n} not divisible by desired grid size {grid_size}")
        
    if m % grid_size != 0:
        raise Exception(f"Number of columns {m} not divisible by desired grid size {grid_size}")
        
    #grid_N * grid_N = number of grids in image
    grid_N = n / grid_size
    grid_M = m / grid_size
    print(f"*{int(grid_N * grid_M)}* number of {grid_size} by {grid_size} grids can fit inside our {n} by {m} image")
    
    grids = []
    for i in range(0, n, grid_size):
        for j in range(0, m, grid_size):
            cur = img_raw[i : i + grid_size, j : j + grid_size]

            grids.append(cur)
            
    if len(grids) != grid_N * grid_M:
        raise Exception("Final length of grids not equal to expected number of grids in image")

    for i in range(len(grids)):
        if show == -1 or i < show:
            print("The " + str(i+1) + "th grid is: ")
            plot_image(grids[i])
        path = os.path.join("gridded_images", name + "_gridded_" + str(i) + ".png")
        cur_img = Image.fromarray(grids[i])
        cur_img.save(path)
        
        
if __name__ == "__main__":
    img = Image.open(os.path.join("test_images", "timothycapybara.png"))
    print(type(img))
    gridImage(img, "timothycapybara")