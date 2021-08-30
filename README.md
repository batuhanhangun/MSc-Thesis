# MSc-Thesis
A repository for the codes used in my MSc thesis.
This thesis contains the CPU and GPU implementations of the method given in https://ieeexplore.ieee.org/document/8071892

## ABSTRACT 
**PERFORMANCE EVALUATION OF A PARALLEL IMAGE ENHANCEMENT TECHNIQUE FOR DARK IMAGES ON MULTITHREADED CPU AND GPU ARCHITECTURES** <br/>

Image processing is a research area with applications in various fields. In time, the complexity of the algorithms and the resolution of the images used in this field have increased. Consequently, single-core central processing units have started to become insufficient. As a solution, researchers have deployed multicore central processing units to accelerate image processing applications. When the multicore central processing units have become inadequate, researchers have started to use graphics processing units. Those devices have hundreds of arithmetic and logic units to speed the image processing applications up. One can also program graphics processing unit using low-level programming interfaces and high-level programming interfaces as with standard microcontrollers. Even though they provide a significant amount of acceleration, low-level interfaces require high development time and deep knowledge about the hardware. For researchers, it is not suitable to spend most of their time on software development. In this thesis, we aimed to show the acceleration capabilities of a high-level programming interface to encourage more image processing researchers to use graphics processing units in their studies, spend less development time, and significantly gain speed up. Within the scope of the study, an image processing method selected from the literature was implemented using the C++ programming language with the OpenMP application programming interface and the CUDA-based OpenCV application programming interface. We first execute the program on a cloud computer with forty-eight cores to measure the performance of multicore central processing units. Then we implemented the same method on a personal computer which has NVIDIA GeForce 1050 GTX TI graphics card and Intel i7-7700HQ central processing unit. Experiments showed that the graphics processing unit provides twenty-five times speedup against the central processing unit depending on some factors.
