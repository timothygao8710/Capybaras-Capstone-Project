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

from importlib import reload
import os
import math

import grid_image_script
import filter_image_script

grid_image_script = reload(grid_image_script)
filter_image_script = reload(filter_image_script)

#########################################################################################################################################################

#Name the image
name = "LowResWomenPortrait"

#Path of image to use
# path = os.path.join("fun", "NoahCapybara.png")
path = os.path.join("test_images", "LowResWomenPortrait.jpeg")
# path = os.path.join("brain_tumor", "braintumor4.jpeg")

#Detection algorithm works with on NxN grids of the original image - limited by # of qubits real quantum computer can sustain
#N is a power of 2
N = 16

#Percent edges for relative thresholding
percent_edges = 0.2
# percent_edges = 0.7

data_qb = math.ceil(math.log2(N**2))
anc_qb = 1
total_qb = data_qb + anc_qb

print(f"This run will require {total_qb} qubits \n")

all_diffs = []
raw_grids = []
#########################################################################################################################################################

def plot_image(img, title: str):
    plt.title(title)
    
    if img.shape[0] <= 16:
        plt.xticks(range(img.shape[0]))
        plt.yticks(range(img.shape[1]))
    
    plt.imshow(img, extent=[0, img.shape[0], img.shape[1], 0])
    plt.show()
    
def save_image(cur_img, path = os.path.join("debug.png")):
    cur_img = cur_img.astype(np.uint8)
    cur_img = Image.fromarray(cur_img)
    if cur_img.mode != 'RGB':
        cur_img = cur_img.convert('RGB')
    cur_img.save(path)

#Normalize -- squared amplitudes must sum to 1
def amplitude_encode(img_data):
    # Calculate the RMS value
    rms = np.sqrt(np.sum(img_data**2))
    #Edge case: All zeroes
    if rms == 0:
        rms = np.sqrt(N * N * 1e-30)
    
    # Create normalized image
    image_norm = []
    for arr in img_data:
        for ele in arr:
            image_norm.append(max(ele, 1e-15) / rms)

    # Return the normalized image as a numpy array
    return np.array(image_norm)

def filter_image(image):
    print("Filtering image...")
    
    image = filter_image_script.enhance_contrast(image)
    if len(image.shape) == 2:
        return image

    #Grey-scale
    return 0.2989 * image[:,:,0] + 0.5870 * image[:,:,1] + 0.1140 * image[:,:,2]

#change backend to IBM backend to run on real quantum computer. Default is simulation
def edge_detection(grid, back=Aer.get_backend('statevector_simulator')):
    global all_diffs
    global raw_grids

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
    all_diffs = np.append(all_diffs, np.array([np.abs(sv_h[2*i+1].real) for i in range(N*N)]))
    all_diffs = np.append(all_diffs, np.array([np.abs(sv_v[2*i+1].real) for i in range(N*N)]))
    raw_grids.append((sv_h, sv_v))

    # # thresh = np.sort(np.array([np.abs(sv_h[2*i+1].real) for i in range(N*N)]))[int(N * N * (1 - percent_edges))]
    # # thresh = np.max(np.array([np.abs(sv_h[2*i+1].real) for i in range(N*N)])) * (1 - percent_edges)
    # print(thresh)

    # # Defining a lambda function for thresholding difference values
    # threshold = lambda amp: (amp > thresh or amp < -1 * thresh)

    # # Selecting odd states from the raw statevector and
    # # reshaping column vector of size N*N to an NxN matrix
    
    # edge_scan_h = np.abs(np.array([1 if threshold(sv_h[2*i+1].real) and (i + 1) % N != 0 else 0 for i in range(N*N)])).reshape(N, N)
    # edge_scan_v = np.abs(np.array([1 if threshold(sv_v[2*i+1].real) and (i + 1) % N != 0 else 0 for i in range(N*N)])).reshape(N, N).T

    # # Plotting the Horizontal and vertical scans
    # # plot_image(edge_scan_h, 'Horizontal scan output')
    # # plot_image(edge_scan_v, 'Vertical scan output')
    
    # # Combining the horizontal and vertical component of the result with bitwise OR
    # edge_scan_sim = edge_scan_h | edge_scan_v

    # # plot_image(edge_scan_sim, 'Edge Detected image')
    # return edge_scan_sim

#########################################################################################################################################################

img_raw = np.asarray(Image.open(path).convert('L'),dtype=int)
print(img_raw.shape)
# img_raw = filter_image(img_raw)
# plot_image(img_raw, name + " Filtered Image")
image_size = int(img_raw.shape[0] * img_raw.shape[1])
tot = int(image_size / N / N)
all_grids = grid_image_script.grid_image(img_raw, name, grid_size=N, show=-2, save_grid_image=False)

for i in range(len(all_grids)):
    # plot_image(all_grids[i], f"{i+1}th Grid for {name}")
    all_grids[i] = all_grids[i].astype(int)
    # cur = edge_detection(all_grids[i])
    edge_detection(all_grids[i])
    
    print(f"Done {i+1}/{tot}...")
    # plot_image(cur, f"{i+1}th Grid for {name}")
    # all_grids[i] = cur

print("Combining grids...")

print(len(all_diffs))
print(image_size)

thresh = np.sort(np.array([np.abs(all_diffs[2*i+1].real) for i in range(image_size)]))[int(image_size * (1 - percent_edges))]
print(thresh)
threshold = lambda amp: (amp > thresh or amp < -1 * thresh)

for i in range(len(all_grids)):
    sv_h = np.asarray(raw_grids[i][0])
    sv_v = np.asarray(raw_grids[i][1])

    edge_scan_h = np.abs(np.array([1 if threshold(sv_h[2*i+1].real) and (i + 1) % N != 0 else 0 for i in range(N*N)])).reshape(N, N)
    edge_scan_v = np.abs(np.array([1 if threshold(sv_v[2*i+1].real) and (i + 1) % N != 0 else 0 for i in range(N*N)])).reshape(N, N).T
    edge_scan_sim = edge_scan_h | edge_scan_v
    all_grids[i] = edge_scan_sim

res_image = grid_image_script.combine_grids(all_grids, img_raw.shape, N)
# plot_image(res_image, name + " Final Combined Image")
save_image(255 - res_image * 255, path = os.path.join("result_images", f"res{100*percent_edges}%{name}.png"))
print("DONE")


# thresh = np.max(np.array([np.abs(sv_h[2*i+1].real) for i in range(image_size)])) * (1 - percent_edges)
