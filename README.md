# To Do #
- Separate single juypter nb into multiple ---> neater and help avoid merge conflicts
- We should have at least a separate part for image preprocessing (what we feed into the algo), quantum part, quantum running on cloud notebooks.
- Could think about adding resource monitoring nb
- Image gridding nb --> large image to gridded smaller images

# Capybaras-Capstone-Project

## Thought process
There were some difficulties in choosing our topic. We initially wanted to do matrix multiplication optimization, but there was a lack of reliable information, so we decided to switch gears.

We decided to implement an edge detection algorithm using the Hadamard gate called QHED (Quantum Hadamard Edge Detection). Our implementation also included QPIE (Quantum Probability Image Encoding) as image preprocessing.  

## Implementation

07/28 - decided on topic and took [notes](https://docs.google.com/document/d/1KwwHY0z-jrOcwBCqH7Xco5jy1c1H4JHYLDq1byuVJ5E/edit?usp=sharing); began implementing some helper functions and finding test images

07/29 - continued implementation

Observed problems:

1. Code has difficulty detecting unclear edges; i.e. when colors are on a gradient. Possible solves: use different threshold values

2. Images that have clear edges with one pixel width of the different color are hard to spot (see small3.png) and the code overlays it on accident

3. Edge detector sometimes has extra pixels in space where there's color and doesn't detect edges when its on vertical ends but does when its on horizontal (see small4.png output).

Goals for Monday:
- Merge most information - sharpening image, large image breakdown
- split code (such as quantum hardware part) into separate notebooks
- do more stuff

## Video
to be uploaded

## Future Steps
- More Evaluation and Benchmarking of QHED performance
- Testing on Randomly Generated images, comparing performance to classical
- Evaluate performance across different image types (e.g., grayscale?)
- Zhang, Yi, Kai Lu, and YingHui Gao. "QSobel: a novel quantum image edge extraction algorithm." Science China Information Sciences 58.1 (2015): 1-13. https://link.springer.com/article/10.1007/s11432-014-5158-9

## Credits
MIT Beaverworks

[Quantum Image Processing](https://journals.aps.org/prx/pdf/10.1103/PhysRevX.7.031041)

[Qiskit Documentation on QHED](https://qiskit.org/textbook/ch-applications/quantum-edge-detection.html)

[8x8 Pixel image creator](https://www.pixilart.com/draw)
