Automatic Discovery of the Statistical Types of Variables in a Dataset
---------------------------------------------------------

This code implements the Bayesian method and reproduces the experiment in 

     I. Valera and Z. Ghahramani, 
     "Automatic Discovery of the Statistical Types of Variables in a Dataset", 
     34th International Conference on Machine Learning (ICML 2017). Sydney (Australia), 2017.

Please, use the above details to cite this work.


Calling from Matlab
-------------------
    function simLik(datasetC,Nits,KK,it)
    %% runs proposed Bayesian method to infer the datatypes in a dataset.
    % Inputs:
    %   datasetC: name of the dataset to be inferred
    %   Nit: number of interations of the Gibbs sampler
    %   KK: low rank representation complexity (i.e., number of features)
    %   itt: number of simulation
    % Outputs: returns void but saves a file with the restuls, i.e., the
    % test log-likelihood adn a vector with the inferred weights for the
    % different datatypes in each dimension.

Alternatively, the fucntion simComp(datasetC,Nits,KK,itt) runs baseline in the paper above, which assumes all the continuous variables to be Gaussian and all the dicrete variables to be categorical

Requirements
------------

    - Matlab 2012b or higher
    - GSL library
        In UBUNTU: sudo apt-get install libgsl0ldbl or sudo apt-get install libgsl0-dev
    - GMP library
        In UBUNTU: sudo apt-get install libgmp3-dev

Installation Instructions
--------------------------

In order to run GLFM on your data, you need to:

1) Download the latest git repository
2) Install necessary libraries:
    - Anaconda: https://www.continuum.io/downloads
    - Library gsl: conda install gsl

3) Compile the C++ code, either for MATLAB or for PYTHON
    - For MATLAB:
        - Add path 'Ccode' (or 'Baseline' to run the baseline) and its children to Matlab workspace
        - From matlab command window, execute:
            mex  -lgsl -lgmp -lgslcblas DataTypes.cpp  (or Baseline.cpp to run the baseline)

-------
Contact
-------

For further information or contact:

    Isabel Valera: isabel.valera.martinez (at) gmail.com


