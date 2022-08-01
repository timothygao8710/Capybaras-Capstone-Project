# Install qiskit
# Install ipywidgets
# Install opencv-python
# Install Pillow
# Install matplotlib
# Install importlib
# Install numpy

# Importing standard Qiskit libraries and configuring account
from qiskit import *
from qiskit import IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from qiskit.tools.monitor import job_monitor
from qiskit.providers.ibmq import least_busy

import cv2
from PIL import Image
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('bmh')

from importlib import reload
import os
import math

import grid_image_script
import filter_image_script

grid_image_script = reload(grid_image_script)
filter_image_script = reload(filter_image_script)

#########################################################################################################################################################

#Name the image
name = "Sunny_Capybara"

#Path of image to use
# path = os.path.join("test_images", "timothycapybara.png")
path = os.path.join("test_images", "test_grey.png")

#Detection algorithm works with on NxN grids of the original image - limited by # of qubits real quantum computer can sustain
#N is a power of 2
N = 8

data_qb = math.ceil(math.log2(N**2))
anc_qb = 1
total_qb = data_qb + anc_qb

print(f"This run will require {total_qb} qubits \n")
#########################################################################################################################################################

def plot_image(img, title: str):
    plt.title(title)
    
    if img.shape[0] <= 16:
        plt.xticks(range(img.shape[0]))
        plt.yticks(range(img.shape[1]))
    
    plt.imshow(img, extent=[0, img.shape[0], img.shape[1], 0], cmap='viridis')
    plt.show()
    
#Normalize -- squared amplitudes must sum to 1
def amplitude_encode(img_data):
    
    # Calculate the RMS value
    rms = np.sqrt(np.sum(np.sum(img_data**2, axis=1)))
    
    # Create normalized image
    image_norm = []
    for arr in img_data:
        for ele in arr:
            image_norm.append(ele / rms)
        
    # Return the normalized image as a numpy array
    return np.array(image_norm)

def filter_image(image):
    print("Filtering image...")
    
    # image = filter_image_script.enhance_contrast(image)
    return image[:,:,0] / 255
    # return image[:,:,1]
    # return image[:,:,2]

#change backend to IBM backend to run on real quantum computer. Default is simulation
def edge_detection(grid, back=Aer.get_backend('statevector_simulator')):
    n = grid.shape[0]
    m = grid.shape[1]

    if n != N or m != N:
        raise Exception("Grid size different from desired grid size")

    # Horizontal: Original image
    image_norm_h = amplitude_encode(grid)

    # Vertical: Transpose of Original image
    image_norm_v = amplitude_encode(grid.T)

    # horizontal scan circuit
    qc_h = QuantumCircuit(total_qb)
    qc_h.initialize(image_norm_h, range(1, total_qb))    

    # vertical scan circuit
    qc_v = QuantumCircuit(total_qb)
    qc_v.initialize(image_norm_v, range(1, total_qb))  

    #Unitary matrix for amplitude permutation
    D2n_1 = np.roll(np.identity(2**total_qb), 1, axis=1)
    
    #QHED for horizontal scan circuit
    qc_h.h(0)
    qc_h.unitary(D2n_1, range(total_qb))
    qc_h.h(0)
    # print(qc_h)
    
    #QHED for vertical scan circuit
    qc_v.h(0)
    qc_v.unitary(D2n_1, range(total_qb))
    qc_v.h(0)
    # print(qc_v)
    
    # Store both circuits in a list, so we can run both circuits in one simulation later
    circ_list = [qc_h, qc_v]
    
    results = execute(circ_list, backend=back).result()
    sv_h = results.get_statevector(qc_h)
    sv_v = results.get_statevector(qc_v)
    
    # Classical postprocessing for plotting the output

    # Defining a lambda function for thresholding difference values
    threshold = lambda amp: (amp > 1e-15 or amp < -1e-15)

    # Selecting odd states from the raw statevector and
    # reshaping column vector of size 64 to an 8x8 matrix
    edge_scan_h = np.abs(np.array([1 if threshold(sv_h[2*i+1].real) else 0 for i in range(2**data_qb)])).reshape(N, N)
    edge_scan_v = np.abs(np.array([1 if threshold(sv_v[2*i+1].real) else 0 for i in range(2**data_qb)])).reshape(N, N).T

    # Plotting the Horizontal and vertical scans
    # plot_image(edge_scan_h, 'Horizontal scan output')
    # plot_image(edge_scan_v, 'Vertical scan output')
    
    # Combining the horizontal and vertical component of the result with bitwise OR
    edge_scan_sim = edge_scan_h | edge_scan_v

    # plot_image(edge_scan_sim, 'Edge Detected image')
    return edge_scan_sim

#########################################################################################################################################################

img_raw = np.asarray(Image.open(path))
plot_image(img_raw, name + " Input Image")
img_raw = filter_image(img_raw)
plot_image(img_raw, name + " Filtered Image")

all_grids = grid_image_script.grid_image(img_raw, name, grid_size=N, show=3, save_grid_image=False)
for i in range(len(all_grids)):
    cur = edge_detection(all_grids[i])
    print(f"{i+1}th Grid Done")
    plot_image(cur, f"{i+1}th Grid for {name}")
    all_grids[i] = cur

res_image = grid_image_script.combine_grids(all_grids, img_raw.shape, N)
plot_image(res_image, name + " Final Combined Image")
