# NIR-data-processing-pipeline
A python based data processing pipeline for NIR spectroscopy

Automatic pipeline to process raw NIR spectrum file (.xlsx)

**Environment requirements:**  
environment.yml

**Files:**  
Run the processing pipeline: main.py  
spectrum file path and formulation info: config.py  
Funtions for processing: NIR_processing.py  
Jupyter notebook example: example.ipynb  


**Pipeline procedure:**
1. Import data, seperate metadata and spectrum data
2. Get metadata info
3. Plot raw spectrum
4. Process raw data using savgol filter smoothing, SNV and mean centering
5. Plot processed spectrum color coded by content (weight percvent of Arginine, Weight percent of Sucrose)
6. Tune PCA hyperparameters by checking explained variance
7. Detect possible outlier in Q residual - Hotelling T^2 plot
8. Draw PCA score plot, colored by selected metadata column
9. Find optimal number of latent variable for PLS
10. Train PLS model with determined number of components
11. Check prediction plot and residual plot of the PLS model

**To run the code:**
1. Change the file path and formulation in config.py
2. Determine the target response and edit accordingly in step 5, 8, 9, 10, 11
3. Run the code in the terminal

**Data resources:**
Raw NIR data from the following publication: https://doi.org/10.1016/j.chemolab.2024.105291













