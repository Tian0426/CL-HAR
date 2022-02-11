
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

## Loss Functions
- NTXent ```models/loss.py```
- Cosine Similarity

## Augmentations
Refer to ```augmentations.py```
- shuffle:channel-wise shuffle
- jitter: add random variations to the signal
- scaling: apply same distortion to the signals from each sensor
- permutation: splitting the signal into a random number of segments with a maximum of M and randomly shuffling them
- jit_scal
- perm_jit
- resample
- na

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
