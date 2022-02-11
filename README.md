# CL-HAR

CL-HAR is an open-source PyTorch library of contrastive learning on wearable-sensor-based human activity recognition (HAR). For more information, please refer to our paper "What Makes Good Contrastive Learning on Small-scale Wearable-based Tasks?".

For more of our results, please refer to [results.md](results.md)

## Installation Requirements
To install required packages, run the following code. The current Pytorch version is 1.8.

```
conda create -n CL-HAR python=3.8.3
conda activate CL-HAR
pip install -r requirements.txt
```
## Supported Datasets
- UCIHAR
- SHAR
- HHAR

## Encoder Networks
Refer to ```models/backbones.py```
- FCN
- DeepConvLSTM
- LSTM
- AE
- CAE
- Transformer

## Contrastive Models
Refer to ```models/models.py```. For sub-modules (projectors, predictors) in the frameworks, refer to ```models/backbones.py```
- TS-TCC 
- SimSiam
- BYOL
- SimCLR
- NNCLR

## Architectures of Contrastive Models
![contrastive_models](figures/contrastive_models.png)

## Architectures of Backbone Networks
![backbone_networks](figures/backbone_networks.png)

## Loss Functions
- NTXent ```models/loss.py```
- Cosine Similarity

## Augmentations
Refer to ```augmentations.py```
### Time Domain
- noise
- scale
- negate
- perm
- shuffle
- t\_flipped
- t\_warp
- resample
- rotation
- perm\_jit
- jit\_scal

### Frequency Domain
- hfc
- lfc
- p\_shift
- ap\_p
- ap\_f


## Utils
- logger
- t-SNE
- MDS

## Reference
- https://github.com/emadeldeen24/TS-TCC
- https://github.com/facebookresearch/simsiam
- https://github.com/lucidrains/byol-pytorch
- https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data
- https://github.com/fastnlp/fitlog
- https://github.com/lightly-ai/lightly
