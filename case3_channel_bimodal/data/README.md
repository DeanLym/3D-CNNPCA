1. Datasets for Case3 should be place in this folder.

2. Datasets are available in folder "case3_bimodal_channel"

https://drive.google.com/drive/folders/1rIH-lWUcYashjVcJkBtCFr4ralOv3sB0?usp=sharing

3. Files in this folder:
    (1) logksand_petrel_train3000_case3.h5
        - 3000 training realizations of log-permeability in sand facies from Petrel
        - Data format: numpy array of shape (nr=3000, nz=40, ny=60, nx=60, 1)
        - File format: hdf5 file with one dataset "data"
    (2) logkmud_petrel_train3000_case3.h5
        - 3000 training realizations of log-permeability in mud facies from Petrel
        - Data format: numpy array of shape (nr=3000, nz=40, ny=60, nx=60, 1)
        - File format: hdf5 file with one dataset "data"
    (3) logksand_petrel_test200_case3.h5
        - 200 test realizations of log-permeability in sand facies from Petrel
        - Data format: numpy array of shape (nr=200, nz=40, ny=60, nx=60, 1)
        - File format: hdf5 file with one dataset "data"
    (4) logkmud_petrel_test200_case3.h5
        - 200 test realizations of log-permeability in mud facies from Petrel
        - Data format: numpy array of shape (nr=200, nz=40, ny=60, nx=60, 1)
        - File format: hdf5 file with one dataset "data"
    (5) facies_petrel_test200_case3.h5
        - 200 test realizations of facies from Petrel
        - Data format: numpy array of shape (nr=200, nz=40, ny=60, nx=60, 1)
        - File format: hdf5 file with one dataset "data"
    (6) facies_cnnpca_test200_case3.h5
        - 200 test realizations of facies from CNN-PCA
        - Data format: numpy array of shape (nr=200, nz=40, ny=60, nx=60, 1)
        - File format: hdf5 file with one dataset "data"
    (7) logksand_mpca_test200_case3.h5
        - 200 new PCA test realizations of log-permeability in sand
        - reduced dimension l=200
        - Data format: numpy array of shape (nr=200, nz=40, ny=60, nx=60, 1)
        - File format: hdf5 file with one dataset "data"
    (6) logkmud_mpca_test200_case3.h5
        - 200 new PCA test realizations of log-permeability in mud
        - reduced dimension l=200
        - Data format: numpy array of shape (nr=200, nz=40, ny=60, nx=60, 1)
        - File format: hdf5 file with one dataset "data"
       
    

