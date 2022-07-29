# Capybaras-Capstone-Project

## Thought process
There were some difficulties in choosing our topic. We initially wanted to do matrix multiplication optimization, but there was a lack of reliable information, so we decided to switch gears.

We decided to implement an edge detection algorithm using the Hadamard gate called QHED (Quantum Hadamard Edge Detection). Our implementation also included QPIE (Quantum Probability Image Encoding) as image preprocessing.  

## Implementation

07/28 - decided on topic and took notes; began implementing some helper functions and finding test images

07/29 - continued implementation

Observed problems:

1. Code has difficulty detecting unclear edges; i.e. when colors are on a gradient. Possible solves: use different threshold values

2. Images that have clear edges with one pixel width of the different color are hard to spot (see small3.png) and the code overlays it on accident

3. Edge detector sometimes has extra pixels in space where there's color and doesn't detect edges when its on vertical ends but does when its on horizontal (see small4.png output).

## Video
to be uploaded

## Credits
MIT Beaverworks

[Qiskit Documentation on QHED](https://qiskit.org/textbook/ch-applications/quantum-edge-detection.html)