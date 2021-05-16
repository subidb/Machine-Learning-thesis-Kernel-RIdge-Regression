

Low-rank approximate machine learning training in kernel ridge regression on GPU

Subid Basaula, Jacobs University

Supervised by Prof. Peter Zaspel


This Jupyter Notebook contains the python code required to run Pivoted Cholesky decomposition(PCD) on both the GPU and the CPU.
Then the GPU version is used to approximate the Kernel Matrix for Kernel Ridge Regression.
Then model training can be done and results can be generated.

References used for Pivoted Cholesky Decomposition:

# H Harbrecht, M Peters, R Schneider. On the low-rank approximation by the pivoted Cholesky decomposition. Applied numerical mathematics, 62(4):428-440, 2012.
#reference : https://github.com/tensorflow/probability/blob/v0.12.2/tensorflow_probability/python/math/linalg.py#L264-L404

Results generated and plotted:

1. Error in Frobenius Norm vs Rank
2. Time taken for GPU vs CPU for PCD, GPU speedup
3. Generalization Error vs Rank of approximated Matrix.

Each code cell is explained by the cell above in the notebook and it is rather intuitive to play around with the parameters and generate results.

Required libraries:

- numpy 
- cupy 
- sklearn
- timeit 
- matplotlib 

All the required libraries are also on the top cell of the notebook:

Each cell can be run in order(with desired data and parameters) to generate the results.










