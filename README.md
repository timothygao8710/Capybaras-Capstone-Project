# Capybaras-Capstone-Project

Hello 👋! We are the Capybaras, from MIT BWSI.

**INSERT VIDEO HERE**

[Slides](https://docs.google.com/presentation/d/1h0SkbEiLvZD52KO8aTaIzguUdenkKmDMlsFrbe9X76Q/edit?usp=sharing)

## Thought process
There were some difficulties in choosing our topic. We initially wanted to do Quantum matrix multiplication optimization, but there was a lack of reliable information, so we decided to switch gears.

We decided to implement an edge detection algorithm using the Hadamard gate called QHED (Quantum Hadamard Edge Detection). Our implementation also included QPIE (Quantum Probability Image Encoding) as image preprocessing.  

## Implementation

07/28 - literature review, decided on topic and took [notes](https://docs.google.com/document/d/1KwwHY0z-jrOcwBCqH7Xco5jy1c1H4JHYLDq1byuVJ5E/edit?usp=sharing); began implementing some helper functions and finding test images

07/29 - implementation, implementation, implementation. Got main algorithm to run without bugs, but fails on many images. Implementing helped us understanding the algorithm fully, and we wrote down explanations for QPIE and QHED in markdown cells.

Observed problems:

1. Code has difficulty detecting unclear edges; i.e. when colors are on a gradient.

2. Images that have clear edges with one pixel width of the different color are hard to spot (see small3.png) and the code overlays it on accident

3. Edge detector sometimes has extra pixels in space where there's color and doesn't detect edges when its on vertical ends but does when its on horizontal (see small4.png output).

Weekend - finished algorithms and notes in jupyter notebook, split files up. Wrote initial code for Image Gridding, improved code for Main.ipynb.

8/1 - continued implementation, began writing script for video. Worked on bugs in large image detection. We found that there were actually bugs with the Qiksit resource (e.g., in amplitude encoding), made code a lot more robust. Started Main_Pipeline.py, also reorganizing the entire repository. Got Image Gridding to work properly. 

8/2 - Got very nice results for some images, but not for others (especially larger images). Found fundamental flaw with thresholding --> Came up with two ways to solving the problem (i.e., K-best and Max Adaptive Thresholding). Started working on slides, continuing to work on script. Fixing many bugs in Main_Pipeline.py.

8/3 - Finished Main_Pipeline.py, running it on a bunch of different images, got some very exciting results. Working on paper. More Debugging. Working on slides and script.

8/4 - Handling some edge cases, running on more images for results. Finalizing paper, slides, script, recording video.

## Future Steps
- Further Evaluation and Benchmarking of QHED performance
- More testing on many for improved robustness (Could use [Google Imagen](https://imagen.research.google/))
- Improved Image filter processes
- Zhang, Yi, Kai Lu, and YingHui Gao. "QSobel: a novel quantum image edge extraction algorithm." Science China Information Sciences 58.1 (2015): 1-13. https://link.springer.com/article/10.1007/s11432-014-5158-9

## Try it yourself!

1. Clone the respository
2. Add image to test_images folder
3. Open Main_Pipeline.py, change path and other settings as desired
4. Run the script
5. Look in result_images, celebrate

## Credits
MIT Beaverworks

[Quantum Image Processing](https://journals.aps.org/prx/pdf/10.1103/PhysRevX.7.031041)

[Qiskit Documentation on QHED](https://qiskit.org/textbook/ch-applications/quantum-edge-detection.html)

[8x8 Pixel image creator](https://www.pixilart.com/draw)

[Cityscapes dataset](https://www.cityscapes-dataset.com/)