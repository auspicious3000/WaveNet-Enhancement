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

### Usage

python bawn_pr_multi_gpu_train /logdir NUM_GPUS

## Train Likelihood Model

Besides the pre-trained prior model, the likelihood model consists of 2 more copies of WaveNet as in the prior model.

Therefore, the clean input shape is 20477 by data size, the noisy input shape is 24570 by data size, and the output shape is 16384 by data size.

Noisy input data is raw noisy audio samples. Clean input data is the bin indices of its clean counterpart. Output data is the expected prediction the same as in training prior model.

Clean input data: clean_train.mat
DataType: int32

Noisy input data: noisy_train.mat
DataType: float32

Output data: target_train.mat
DataType: uint8

### Usage

python bawn_ll_multi_gpu_train.py /logdir /path_to_prior_model NUM_GPUS

