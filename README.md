# WaveNet-Enhancement

bawn.py contains contains most of the functions including definition of the WaveNet structure.

bawn_pr_multi_gpu_train.py trains the prior model. 

bawn_ll_multi_gpu_train.py trains the likelihood model with a fixed prior model.

## Train Prior Model 

The prior model is a 40 layer WaveNet consists of 4 blocks with 10 layers each.

Therefore, the input shape is 20477 by data size, and the output shape is 16384 by data size.

Input data is bin indices of clean speech using 256 mu-law quantization.
Output data is the corresponding prediction shifted by one sample to the right, making the model always predicts the next sample based on the past samples.

Input data: train_pr.mat     
DataType: int32
Output data: target_pr.mat
DataType: uint8

