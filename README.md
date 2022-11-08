![BWSI Results](https://github.com/timothygao8710/Capybaras-Capstone-Project/blob/main/BWSIResults.gif)

# Capybaras-Capstone-Project
<sub> Timothy Gao, Tina Wang, Cassiel Graullera, Noah Cooney

Hello 👋! We are the Capybaras, from MIT BWSI.

[Video Link](https://www.youtube.com/watch?v=93m1npEH2nw)

[Slides Link](https://docs.google.com/presentation/d/1h0SkbEiLvZD52KO8aTaIzguUdenkKmDMlsFrbe9X76Q/edit?usp=sharing)

We decided to implement an edge detection algorithm using the Hadamard gate called QHED (Quantum Hadamard Edge Detection). Our implementation also included QPIE (Quantum Probability Image Encoding) as image preprocessing.  

## Project Timeline

07/27 - Literature review, searching for topic. We initially decided on Quantum matrix multiplication optimization (and use it to speed-up finding the Nth k-fibonacci number), but we quickly realized that there was a lack of reliable information, so we decided to switch gears, and landed on Quantum Edge Detection.

07/28 - More Literature Review, taking [notes](https://docs.google.com/document/d/1KwwHY0z-jrOcwBCqH7Xco5jy1c1H4JHYLDq1byuVJ5E/edit?usp=sharing); began implementing some helper functions and finding test images

07/29 - implementation, implementation, implementation. Got main algorithm to run without bugs, but fails on many images. Implementing helped us understanding the algorithm fully, and we wrote down explanations for QPIE and QHED in markdown cells.

Observed problems:

1. Code has difficulty detecting unclear edges; i.e. when colors are on a gradient.

2. Images that have clear edges with one pixel width of the different color are hard to spot (see small3.png) and the code overlays it on accident

3. Edge detector sometimes has extra pixels in space where there's color and doesn't detect edges when its on vertical ends but does when its on horizontal (see small4.png output).

Weekend - finished algorithms and notes in jupyter notebook, split files up. Wrote initial code for Image Gridding, improved code for Main.ipynb.

8/1 - continued implementation, began writing script for video. Worked on bugs in large image detection. We found that there were actually bugs with the Qiksit resource (e.g., in amplitude encoding), made code a lot more robust. Started Main_Pipeline.py, also reorganizing the entire repository. Got Image Gridding to work properly. 

8/2 - Got very nice results for some images, but not for others (especially larger images). Found fundamental flaw with thresholding --> Came up with two ways to solving the problem (i.e., K-best and Max Adaptive Thresholding). Started working on slides, continuing to work on script. Fixing many bugs in Main_Pipeline.py.

8/3 - Finished Main_Pipeline.py, running it on a bunch of different images, got some very exciting results. Working on paper. More Debugging. Working on slides and script.

8/4 & Weekend - Handling some edge cases, running on more images for results. Finalizing paper, slides, script, recording video.

8/7 - Presented our work to over , including the BWSI program

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
5. The result should be in the folder result_images

## Credits

Thank you to our incredible teachers Richard Preston and Joe Clapis (who also developed [the curriculum](https://stem.mitre.org/quantum/index.html)) and our TAs Nikita Borisov, Diptanshu Sikdar, Melvin Lin, Dylan VanAllen, and Jon Christie.

Thank you to MIT and Lincoln Labs for providing us with the opportunity to participate in the [BWSI Beaverworks Summer Institute](https://beaverworks.ll.mit.edu/CMS/bw/bwsi_quantum_software).

![BWSI Logo](https://beaverworks.ll.mit.edu/CMS/bw/sites/all/themes/professional_theme/logo.png)

## Sources
- [Quantum Image Processing and Its Application to Edge Detection: Theory and Experiment](https://journals.aps.org/prx/pdf/10.1103/PhysRevX.7.031041)
- [Quantum Hadamard Edge Detection Algorithm](https://arxiv.org/pdf/2012.11036.pdf)
- [Pixel Image Drawing Tool](https://www.pixilart.com/draw)
- [Qiskit Documentation](https://qiskit.org/textbook/ch-applications/quantum-edge-detection.html)
- [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
- [Google Images](https://images.google.com/)
