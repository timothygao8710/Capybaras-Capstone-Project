import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import os
import math

#raw_image: [np array] Image as np array
#name: [string] name of the image which will be used
#grid_size: [int] desired grid size
#show: [Int] number of gridded images to show, -1 to show all images
#save_grid_image: [Bool] if true, saves each gridded image in the gridded_images folder

def grid_image(img_raw, name, grid_size = 8, show = -1, save_grid_image = False):   
    print("Gridding image...") 
    #Imports
    import matplotlib.image as mpimg
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import style
    import os
    import math
    style.use('bmh')

    plot_image(img_raw)
    # print(img_raw.shape)
    n = img_raw.shape[0]
    m = img_raw.shape[1]

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
        if save_grid_image:
            path = os.path.join("gridded_images", name + "_gridded_" + str(i) + ".png")
            cur_img = Image.fromarray(grids[i])
            cur_img.save(path)
    
    return grids
        
# grids: [list of 2-D np arrays] list of grids to combine
# original_shape: [tuple] original_image.shape
# grid_size: [int] N side length of grid
def combine_grids(grids, original_shape, grid_size):
    res_image = np.zeros(shape=original_shape)

    idx = 0
    for i in range(0, original_shape[0], grid_size):
        for j in range(0, original_shape[1], grid_size):
            res_image[i : i + grid_size, j : j + grid_size] = grids[idx]
            idx += 1

    return res_image

def plot_image(img, title=""):
    plt.title(title)
    
    if img.shape[0] <= 32:
        plt.xticks(range(img.shape[0]))
        plt.yticks(range(img.shape[1]))
    
    plt.imshow(img, extent=[0, img.shape[0], img.shape[1], 0], cmap='viridis')
    plt.show()

if __name__ == "__main__":
    # img = Image.open(os.path.join("test_images", "timothycapybara.png"))
    img = Image.open(os.path.join("test_grey.jpg"))
    img = np.asarray(img)
    print(type(img))
    # gridImage(img, "timothycapybara")
    grids = grid_image(img, "test_grey", 32)
    fin = combine_grids(grids, img.shape, 32)

    plot_image(fin)

    