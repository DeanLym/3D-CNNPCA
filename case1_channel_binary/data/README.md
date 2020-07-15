##### Datasets for Case1 should be place in this folder.

##### Datasets are available in folder "case1\_binary\_channel"

https://drive.google.com/drive/folders/1rIH-lWUcYashjVcJkBtCFr4ralOv3sB0?usp=sharing

##### Files in this folder:
* hard\_data\_case1.pickle
  + Hard data at well locations
  + Data format: python dictionary
  + File format: binary from pickle.dump  
* m\_petrel\_train3000\_case1.h5  
  + 3000 realizations from object-based modelling in Petrel  
  + Data format: numpy array of shape (nr=3000, nz=40, ny=60, nx=60, 1)  
  + File format: hdf5 file with one dataset "data"  
* m\_pca\_rec\_train3000\_case1.h5  
  + 3000 PCA realizations corresponding to m\_petrel  
  + reduced dimension l=400  
  + perturbation apply to xi\_j, j=31,...,400  
  + Data format: numpy array of shape (nr=3000, nz=40, ny=60, nx=60, 1)  
  + File format: hdf5 file with one dataset "data"  
* m\_pca\_train3000\_case1.h5  
  + 3000 new PCA realizations
  + reduced dimension l=400
  + Data format: numpy array of shape (nr=3000, nz=40, ny=60, nx=60, 1)
  + File format: hdf5 file with one dataset "data"  
* m\_petrel\_test200\_case1.h5  
  + 200 new Petrel test realizations   
  + Data format: numpy array of shape (nr=200, nz=40, ny=60, nx=60, 1)  
  + File format: hdf5 file with one dataset "data"  
* m\_pca\_test200\_case1.h5  
  + 200 new PCA test realizations   
  + reduced dimension l=400  
  + Data format: numpy array of shape (nr=200, nz=40, ny=60, nx=60, 1)  
  + File format: hdf5 file with one dataset "data"  
       
##### To use this dataset, either:  
* download all files and start directly with Step3\_CNNPCA\_Train\_Case1.ipynb  
* or   
  + download m\_petrel\_train3000\_case1.h5   
  + construct PCA, generate PCA models using Step1\_Construct\_PCA\_Case1.ipynb,   
  + generate hard data file using Step2\_Prepare\_Hard\_Data\_Case1.ipynb  
  + start training with Step3\_CNNPCA\_Train\_Case1.ipynb  

