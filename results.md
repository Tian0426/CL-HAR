## Visualization of Time-Domain Augmentations (UCIHAR dataset)
![time_aug_heatmap](figures/time_aug_heatmap.png)

## Visualization of Frequency-Domain Augmentations (SHAR dataset)
![freq_aug_heatmap](figures/freq_aug_heatmap.png)

## Augmentation Results on BYOL 
UCIHAR             |  SHAR 
:-------------------------:|:-------------------------:
![byol_aug_ucihar](figures/byol_aug_ucihar.png)  |  ![byol_aug_shar_fcn](figures/byol_aug_shar_fcn.png)

## Augmentation Results on SimSiam 
UCIHAR             |  SHAR 
:-------------------------:|:-------------------------:
![simsiam_aug_ucihar](figures/simsiam_aug_ucihar.png)  |  ![simsiam_aug_shar_fcn](figures/simsiam_aug_shar_fcn.png)

## Augmentation Results on SimCLR
UCIHAR             |  SHAR 
:-------------------------:|:-------------------------:
![simclr_aug_ucihar](figures/simclr_aug_ucihar.png)  |  ![simclr_aug_shar_fcn](figures/simclr_aug_shar_fcn.png)

## Augmentation Results on NNCLR
UCIHAR             |  SHAR 
:-------------------------:|:-------------------------:
![nnclr_aug_ucihar](figures/nnclr_aug_ucihar.png)  |  ![nnclr_aug_shar_fcn](figures/nnclr_aug_shar_fcn.png)

## Augmentation Results on TS-TCC
UCIHAR             |  SHAR 
:-------------------------:|:-------------------------:
![tstcc_aug_ucihar](figures/tstcc_aug_ucihar.png)  |  ![nnclr_aug_shar_fcn](figures/tstcc_aug_shar.png)


The settings listed in the table below achieve reasonable performance. However, the actual hyper-parameter setup should be subject to the changes of backbone networks and augmentations.

| Dataset  | Model   | Learning rate | Batch Size | Weight Decay | Temperature | EMA_decay  | Memory Bank Size | Epoch |
|----------|---------|---------------|------------|--------------|-------------|------------|------------------|-------|
| UCIHAR   | BYOL    | 5e-4          | 128        | 1.5e-6       | -           | 0.996      | -                | 60    |
|          | SimSiam | 5e-4          | 128        | 1e-4         | -           | -          | -                | 60    |
|          | SimCLR  | 3e-3          | 256        | 1e-6         | 0.1         | -          | -                | 120   |
|          | NNCLR   | 3e-3          | 256        | 1e-6         | 0.1         | -          | 1024             | 120   |
|          | TS-TCC  | 3e-4          | 128        | 3e-4         | 0.2         | -          | -                | 40    |
| SHAR     | BYOL    | 1e-3          | 64         | 1.5e-6       | -           | 0.996      | -                | 60    |
|          | SimSiam | 3e-4          | 256        | 1e-4         | -           | -          | -                | 60    |
|          | SimCLR  | 2.5e-3        | 256        | 1e-6         | 0.1         | -          | -                | 120   |
|          | NNCLR   | 2e-3          | 256        | 1e-6         | 0.1         | -          | 1024             | 120   |
|          | TS-TCC  | 3e-4          | 128        | 3e-4         | 0.2         | -          | -                | 40    |
