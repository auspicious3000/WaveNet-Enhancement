from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

import tensorflow as tf
import numpy as np, h5py


#Basic constants
USE_FP16 = False
DATA_DIR = '/scratch/data'
BATCH_SIZE = 1

# Global constants describing the BAWN data set.
LEN_OUTPUT = 16384
LEN_PAD = 4093
NUM_BLOCKS_CLEAN = 4
NUM_LAYERS_CLEAN = 10
NUM_BLOCKS_NOISY = 4
NUM_LAYERS_NOISY = 10
NUM_CLASSES = 256
NUM_POST_LAYERS = 2
NUM_RESIDUAL_CHANNELS_CLEAN = 64
NUM_RESIDUAL_CHANNELS_NOISY = 64
NUM_SKIP_CHANNELS = 256



# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    #print(tensor_name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
    
    
def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if USE_FP16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var 


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    dtype = tf.float16 if USE_FP16 else tf.float32
    var = _variable_on_cpu(
          name,
          shape,
          tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _conv1d(inputs, 
            out_channels, 
            filter_width=2, 
            stride=1, 
            padding='valid',
            data_format='channels_first',
            dilation_rate=1,
            gain=np.sqrt(2), 
            activation=None, 
            bias=True,
            name='',
            trainable=True):
     
    in_channels = inputs.get_shape().as_list()[-2]
    stddev = gain / np.sqrt(filter_width ** 2 * in_channels)
    w_init = tf.random_normal_initializer(stddev=stddev)      #weight matrix init
        
    outputs = tf.layers.conv1d(inputs=inputs, 
                               filters=out_channels, 
                               kernel_size=filter_width, 
                               strides=1, 
                               padding=padding, 
                               data_format=data_format, 
                               dilation_rate=dilation_rate, 
                               activation=activation, 
                               use_bias=bias, 
                               kernel_initializer=w_init, 
                               bias_initializer=tf.zeros_initializer(), 
                               kernel_regularizer=None, 
                               bias_regularizer=None, 
                               activity_regularizer=None, 
                               trainable=trainable, 
                               name=name, 
                               reuse=None)
       
    return outputs



def _embed_conv1d(inputs, out_channels, activation=tf.tanh, bias=True):
    """Compute initial 2x1 conv using embedding
    Args:
      inputs: int32 ids returned from inputs
    Returns:
      output.
    """
    
    #inputs = tf.squeeze(inputs, axis=-2, name='remove_channel')
    
    with tf.variable_scope('pre') as scope:
        ww = _variable_with_weight_decay('kernel',
                                         shape=[256,out_channels,2],
                                         stddev=0.707,
                                         wd=None)
        
        embedded_kernel = tf.nn.embedding_lookup(ww, inputs, name='embedded_kernel')
        
        output = tf.add(embedded_kernel[:,0:-1,:,0], embedded_kernel[:,1:,:,1], name='shift_and_add')
        
        #transform into "channel first"
        output = tf.transpose(output, perm=[0,2,1], name='channel_first')
        
        if bias:
            len_in = inputs.get_shape().as_list()[1]
            b = _variable_on_cpu('bias', [len_in-1], tf.constant_initializer(0.0))
            output = tf.nn.bias_add(output, b, name='add_bias')
        if activation:
            output = activation(output, name='tanh')
    
    return  output



def _dilated_conv1d(inputs,
                    residual_channels,
                    skip_channels, 
                    skip_width, 
                    filter_width=2, 
                    rate=1, 
                    gain=np.sqrt(2),
                    padding='valid', 
                    bias=True,
                    causality=None,
                    trainable=True,
                    bottom=False,
                    top=False,
                    name=None):       
    """

    Args:
      inputs: (tensor)
      residual_channels: num channels at the gated output after 1x1 convolution
      skip_channels: num channels at the skip output after 1x1 convolution
      skip_width: width of the skip output, equivalent to valid width
      filter_width:
      rate:
      padding:
      name:
      gain:
      causality:

    Outputs:
      outputs: (tensor)

    Implements the following architecture
       input (size = in_channel) -> 1x1 conv (optional) -> input_proc(size=out_channel) -...

                               |-> [gate]   -|                                 |-> 1x1 conv -> skip output (size = skip_channel)
                               |             |-> (*)(size = residual_channel) -|
                              -|-> [filter] -|                                 |-> 1x1 conv -|                                                                              |                                                             |-> (+) -> dense output (size=residual_channel)
                               |-------------------------------------------------------------|
    """
    
    assert name
    with tf.variable_scope(name):
        #if layer input and layer residual output have different channels
        #apply a pre-dense layer to input to match No. of channels
        #so that layer residual output and layer input can be added
        if bottom:
            #print(inputs.shape)
            #inputs_proc = _embed_conv1d(inputs, residual_channels)
            if 'noisy' in name:
                inputs = tf.expand_dims(inputs, axis=1, name='add_channel')
            else:
                inputs = tf.one_hot(inputs, 256, axis=1, dtype=tf.float32)
                
            inputs_proc = _conv1d(inputs, 
                                  out_channels=residual_channels, 
                                  filter_width=filter_width, 
                                  padding=padding, 
                                  gain=gain, 
                                  activation=tf.tanh,
                                  bias=bias,
                                  name='pre',
                                  trainable=trainable)
            #print(inputs_proc.shape)
        else:
            # note: it is not appended to hs!!!!!!!!!!!!!!!!!!
            inputs_proc = inputs                             
        
        #replace 2 separate conv with 1 conv with 2x residual channels
        outputs_together = _conv1d(inputs_proc, 
                                   out_channels=2*residual_channels,
                                   filter_width=filter_width, 
                                   padding=padding,
                                   dilation_rate=rate,
                                   gain=gain,
                                   activation=None,
                                   bias=False,
                                   name='together',
                                   trainable=trainable)
    
        
        #slice 2r channels into two r channels
        outputs_filter = tf.slice(outputs_together, [0,0,0], [-1,residual_channels,-1], name='filter_part')
        outputs_gate = tf.slice(outputs_together, [0,residual_channels,0], [-1,-1,-1], name='gate_part')
        
        width = outputs_together.get_shape().as_list()[-1]
                
        #if bias:
        #    bf = _variable_on_cpu('filter/bias', [residual_channels], tf.constant_initializer(0.0))
        #    bg = _variable_on_cpu('gate/bias', [residual_channels], tf.constant_initializer(0.0))
        #    outputs_filter = tf.nn.bias_add(outputs_filter, bf, 'NCHW', name='add_bias_filter')
        #    outputs_gate = tf.nn.bias_add(outputs_gate, bg, 'NCHW', name='add_bias_gate')
        
        #add activations
        outputs_filter = tf.tanh(outputs_filter, name='filter')
        outputs_gate = tf.sigmoid(outputs_gate, name='gate')
        
        outputs_gated = tf.multiply(outputs_filter, outputs_gate, name='filter_X_gate')
             
                
        #remove part of the padding so that the length of the input is 
        #equal to that of the outputs_residual
        assert causality in ['clean','prior','noisy']
        if causality in ['clean','prior']:
            ind_in = np.s_[rate:]
            ind_out = np.s_[-skip_width:]
            width_dense = width - rate
            
        elif causality == 'noisy':
            ind_in = np.s_[rate:-rate]
            #this may not be a multiple of 2 ?????
            len_cut = int((width - skip_width) / 2)
            if len_cut == 0:
                ind_out = np.s_[0:]
            else:
                ind_out = np.s_[len_cut:-len_cut]
            width_dense = width - 2*rate
        
        #apply 1x1 convolution layer to make skip
        #slice only the skipwidth part of the gated output    
        outputs_skip = _conv1d(outputs_gated[:, :, ind_out],
                               out_channels=skip_channels,
                               filter_width=1, 
                               padding=padding, 
                               gain=1, 
                               activation=None,
                               bias=bias,
                               name='skip',
                               trainable=trainable)
    
            
        if not top:
            #apply 1x1 convolution layer to make residual
            outputs_residual = _conv1d(outputs_gated, 
                                       out_channels=residual_channels, 
                                       filter_width=1, 
                                       padding=padding,
                                       gain=1, 
                                       activation=None,
                                       bias=bias,
                                       name='residual',
                                       trainable=trainable)
       
            
            #output dense is residual + input
            outputs_dense = tf.add(inputs_proc[:, :, ind_in], outputs_residual, name='add_residual')
        else:
            outputs_dense = None
    
    #_activation_summary(inputs_proc)
    #_activation_summary(outputs_together)
    #_activation_summary(outputs_filter)    
    #_activation_summary(outputs_gate)        
    #_activation_summary(outputs_gated)
    #_activation_summary(outputs_skip)
    #if not top:
    #    _activation_summary(outputs_residual)                     
    #    _activation_summary(outputs_dense)
     
    return (outputs_dense, outputs_skip, inputs_proc)


def check_boundries(num_blocks, num_layers, block, layer):
    bottom = False
    top = False
    if block==0 and layer==0:
        bottom = True
    if block+1==num_blocks and layer+1==num_layers:
        top = True
    
    return (bottom, top)


def _wavnet(inputs
            ,num_blocks
            ,num_layers
            ,num_residual_channels
            ,num_skip_channels
            ,len_output
            ,filter_width
            ,speech_type
            ,bias
            ,trainable=True):
    
    h = inputs
    hs = []
    hs.append(h[:,-2:-1])
    skips = []
    #with tf.variable_scope(speech_type):
    for b in range(num_blocks):
        for i in range(num_layers):
            rate = 2 ** i
            name = '{}/b{}-l{}'.format(speech_type, b, i)    #layer i of block b
            bottom, top = check_boundries(num_blocks, num_layers, b, i)
            
            h, skip, pre = _dilated_conv1d(h, 
                                           num_residual_channels, 
                                           num_skip_channels, 
                                           len_output, 
                                           filter_width,
                                           rate=rate, 
                                           bias=bias, 
                                           causality=speech_type, 
                                           trainable=trainable,
                                           bottom=bottom,
                                           top=top,
                                           name=name)
            if bottom:
                hs.append(pre)
                hs.append(h)
            elif not top:
                hs.append(h)
            skips.append(skip)
    
    
    return (hs, skips)


def _post_processing(inputs, num_layers, num_classes, name, reuse=False, bias=True, trainable=True):
    """ 
    Performs post-processing (fully connected layers, 1 X 1 convolutions)
    
    inputs: a list of skip outputs of each dialted layer
    num_layers: number of dense layers, including the final output
    num_classes: the dimension of the final output 
    
    """
    #combine all skip outputs and pass through relu
    inputs_agg = tf.add_n(inputs, name='sum_skips')
    h = tf.nn.relu(inputs_agg)
    
    for l in range(num_layers-1):
        with tf.variable_scope('{}post_l{}'.format(name, l), reuse=reuse):
            h = _conv1d(h, 
                        out_channels=num_classes, 
                        filter_width=1, 
                        padding='valid', 
                        gain=1, 
                        activation=tf.nn.relu,
                        bias=bias,
                        trainable=trainable)
        #_activation_summary(h)
            
    #procees the last layer separately because it has no activation
    with tf.variable_scope('{}post_l{}'.format(name, num_layers-1), reuse=reuse):
        outputs = _conv1d(h,
                          out_channels=num_classes, 
                          filter_width=1, 
                          padding='valid', 
                          gain=1, 
                          activation=None,
                          bias=bias,
                          trainable=trainable)
    #_activation_summary(outputs)
        
    return outputs


def _post_processing_batch(inputs, num_layers, num_classes, name, reuse=False, bias=True, trainable=True):
    """ 
    Performs post-processing (fully connected layers, 1 X 1 convolutions)
    
    inputs: a list of gated outputs of each dialted layer
    num_layers: number of dense layers, including the final output
    num_classes: the dimension of the final output 
    
    """
    batch_size, in_channels, width = inputs[0].get_shape().as_list()
    inputs = tf.stack(inputs, axis=1, name='list2tensor')
    inputs_agg = tf.reshape(inputs, [batch_size, -1, width], name='skips_batch')
    with tf.variable_scope('{}skips_all'.format(name), reuse=reuse):
        h = _conv1d(inputs_agg, 
                    out_channels=num_classes,
                    filter_width=1, 
                    padding='valid', 
                    gain=1, 
                    activation=tf.nn.relu,
                    bias=bias,
                    name='skip',
                    trainable=trainable)
    
    for l in range(num_layers-1):
        with tf.variable_scope('{}post_l{}'.format(name, l), reuse=reuse):
            h = _conv1d(h, 
                        out_channels=in_channels, 
                        filter_width=1, 
                        padding='valid', 
                        gain=1, 
                        activation=tf.nn.relu,
                        bias=bias,
                        trainable=trainable)
        #_activation_summary(h)
            
    #procees the last layer separately because it has no activation
    with tf.variable_scope('{}post_l{}'.format(name, num_layers-1), reuse=reuse):
        outputs = _conv1d(h,
                          out_channels=num_classes, 
                          filter_width=1, 
                          padding='valid', 
                          gain=1, 
                          activation=None,
                          bias=bias,
                          trainable=trainable)
    #_activation_summary(outputs)
        
    return outputs



def load_data_prior(train, target):
    
    dest_directory = DATA_DIR
    assert os.path.exists(dest_directory)
    filepath_train = os.path.join(dest_directory, train)
    filepath_target = os.path.join(dest_directory, target)
    
    with h5py.File(filepath_train,'r') as f:
        inputs = np.array(f.get(os.path.splitext(train)[0]))
        
    with h5py.File(filepath_target,'r') as f:
        labels = np.array(f.get(os.path.splitext(target)[0]))
        
    assert (inputs.dtype == 'int32' and labels.dtype == 'int32'), 'Data type incorrect!!!!'
    assert inputs.shape[0] == labels.shape[0], 'The first dimension (batch size) must equal!!!!'    
        
    return (inputs, labels)


def load_data_simple(train, target):
    
    dest_directory = DATA_DIR
    assert os.path.exists(dest_directory)
    filepath_train = os.path.join(dest_directory, train)
    filepath_target = os.path.join(dest_directory, target)
    
    with h5py.File(filepath_train,'r') as f:
        inputs = np.array(f.get(os.path.splitext(train)[0]))
        
    with h5py.File(filepath_target,'r') as f:
        labels = np.array(f.get(os.path.splitext(target)[0]))
        
    assert (inputs.dtype == 'float32' and labels.dtype == 'int32'), 'Data type incorrect!!!!'
    assert inputs.shape[0] == labels.shape[0], 'The first dimension (batch size) must equal!!!!'    
        
    return (inputs, labels)


def data_initializer_prior(data_segments, data_labels):
    # Input data
    segments_initializer = tf.placeholder_with_default(
        tf.zeros(data_segments.shape, tf.int32),
        shape=data_segments.shape,
        name='segments_initializer')
    labels_initializer = tf.placeholder_with_default(
        tf.zeros(data_labels.shape, tf.int32),
        shape=data_labels.shape,
        name='labels_initializer')
    input_segments = tf.Variable(
          segments_initializer, trainable=False, 
        collections=[tf.GraphKeys.LOCAL_VARIABLES], name='input_segments')
    input_labels = tf.Variable(
          labels_initializer, trainable=False, 
        collections=[tf.GraphKeys.LOCAL_VARIABLES], name='input_labels')    
    
    return (segments_initializer, labels_initializer, input_segments, input_labels)


def data_initializer_simple(data_segments, data_labels):
    # Input data
    segments_initializer = tf.placeholder_with_default(
        tf.zeros(data_segments.shape, tf.float32),
        shape=data_segments.shape,
        name='segments_initializer')
    labels_initializer = tf.placeholder_with_default(
        tf.zeros(data_labels.shape, tf.int32),
        shape=data_labels.shape,
        name='labels_initializer')
    input_segments = tf.Variable(
          segments_initializer, trainable=False, 
        collections=[tf.GraphKeys.LOCAL_VARIABLES], name='input_segments')
    input_labels = tf.Variable(
          labels_initializer, trainable=False, 
        collections=[tf.GraphKeys.LOCAL_VARIABLES], name='input_labels')    
    
    return (segments_initializer, labels_initializer, input_segments, input_labels)


def inputs_batch_prior(input_segments, input_labels):
    """Construct input for CIFAR training using the Reader ops.
    Returns:
      segments: 3D tensor of [batch_size, NUM_CHANNELS, LEN_INPUT_CLEAN] size.
      lables: 3D tensor of [batch_size, NUM_CHANNELS, LEN_OUTPUT] size.
    Raises:
    """
    
    #print(input_segments.shape)
    #print(input_labels.shape)
    segment, label = tf.train.slice_input_producer([input_segments, input_labels])
    #label = tf.cast(label, tf.int32)
    #print(segment.shape)
    #print(label.shape)
    segments, labels = tf.train.batch([segment, label], batch_size=BATCH_SIZE)    
        
    if USE_FP16:
        segments = tf.cast(segments, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return segments, labels



def load_data_likli(clean, noisy, target):
    
    dest_directory = DATA_DIR
    assert os.path.exists(dest_directory)
    filepath_clean = os.path.join(dest_directory, clean)
    filepath_noisy = os.path.join(dest_directory, noisy)
    filepath_target = os.path.join(dest_directory, target)
    
    with h5py.File(filepath_clean,'r') as f:
        inputs_clean = np.array(f.get(os.path.splitext(clean)[0]))
        
    with h5py.File(filepath_noisy,'r') as f:
        inputs_noisy = np.array(f.get(os.path.splitext(noisy)[0]))    
        
    with h5py.File(filepath_target,'r') as f:
        labels = np.array(f.get(os.path.splitext(target)[0]))
        
    assert (inputs_clean.dtype == 'int32' and inputs_noisy.dtype == 'float32' and labels.dtype == 'int32'), 'Data type incorrect!!!!'
    assert inputs_clean.shape[0] == inputs_noisy.shape[0] == labels.shape[0], 'The first dimension (batch size) must equal!!!!'    
        
    return (inputs_clean, inputs_noisy, labels)


def data_initializer_likli(data_clean, data_noisy, data_labels):
    # Input data
    clean_initializer = tf.placeholder_with_default(
        tf.zeros(data_clean.shape, tf.int32),
        shape=data_clean.shape,
        name='clean_initializer')
    noisy_initializer = tf.placeholder_with_default(
        tf.zeros(data_noisy.shape, tf.float32),
        shape=data_noisy.shape,
        name='noisy_initializer')
    labels_initializer = tf.placeholder_with_default(
        tf.zeros(data_labels.shape, tf.int32),
        shape=data_labels.shape,
        name='labels_initializer')
    # variables
    input_clean = tf.Variable(
          clean_initializer, trainable=False, 
        collections=[tf.GraphKeys.LOCAL_VARIABLES], name='input_clean')
    input_noisy = tf.Variable(
          noisy_initializer, trainable=False, 
        collections=[tf.GraphKeys.LOCAL_VARIABLES], name='input_noisy')
    input_labels = tf.Variable(
          labels_initializer, trainable=False, 
        collections=[tf.GraphKeys.LOCAL_VARIABLES], name='input_labels')    
    
    return (clean_initializer, noisy_initializer, labels_initializer, input_clean, input_noisy, input_labels)


def inputs_batch_likli(input_clean, input_noisy, input_labels):
    """Construct input for training using the Reader ops.
    Returns:
      segments: 3D tensor of [batch_size, NUM_CHANNELS, LEN_INPUT_CLEAN] size.
      lables: 3D tensor of [batch_size, NUM_CHANNELS, LEN_OUTPUT] size.
    Raises:
    """
    
    clean, noisy, label = tf.train.slice_input_producer([input_clean, input_noisy, input_labels])
    cleans, noisys, labels = tf.train.batch([clean, noisy, label], batch_size=BATCH_SIZE)    
        
    if USE_FP16:
        cleans = tf.cast(cleans, tf.float16)
        noisys = tf.cast(noisys, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return cleans, noisys, labels



def inputs_batch_process(input_clean, input_noisy, batch_size=1):
    """Pre-process input for training using the Reader ops.
    Returns:
      segments: 3D tensor of [batch_size, NUM_CHANNELS, LEN_INPUT_CLEAN] size.
      lables: 3D tensor of [batch_size, NUM_CHANNELS, LEN_OUTPUT] size.
    Raises:
    """
    
    clean, noisy = tf.train.slice_input_producer([input_clean, input_noisy])
    cleans, noisys = tf.train.batch([clean, noisy], batch_size=batch_size)    
        
    if USE_FP16:
        cleans = tf.cast(cleans, tf.float16)
        noisys = tf.cast(noisys, tf.float16)
        
    return cleans, noisys



def model_prior(inputs_clean):
    """Build the BaWN prior model.
    Args:
      inputs_clean: clean audio segment returned from inputs_clean().
    Returns:
      Logits.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #
    # 
    #prior speech model
    _, skips_prior_batch = _wavnet(inputs=inputs_clean,
                                   num_blocks=NUM_BLOCKS_CLEAN, 
                                   num_layers=NUM_LAYERS_CLEAN, 
                                   num_residual_channels=NUM_RESIDUAL_CHANNELS_CLEAN, 
                                   num_skip_channels=NUM_SKIP_CHANNELS, 
                                   len_output=LEN_OUTPUT, 
                                   filter_width = 2,
                                   speech_type='prior',
                                   bias = True,
                                   trainable=True)
    
    outputs_pr_batch = _post_processing(skips_prior_batch, 
                                        NUM_POST_LAYERS, 
                                        NUM_CLASSES, 
                                        'prior/', 
                                        trainable=True)
           
    return outputs_pr_batch


def model_denoise(inputs_clean, inputs_noisy):
    """Build the BaWN denoise model.
    Args:
      inputs_clean: clean audio segment returned from inputs_batch_likelihood().
      inputs_noisy: noisy audio segment returned from inputs_batch_likelihood().
    Returns:
      Logits.
    """
    
    #noise model                
    _, skips_clean_batch = _wavnet(inputs=inputs_clean,
                                  num_blocks=NUM_BLOCKS_CLEAN, 
                                  num_layers=NUM_LAYERS_CLEAN, 
                                  num_residual_channels=NUM_RESIDUAL_CHANNELS_CLEAN, 
                                  num_skip_channels=NUM_SKIP_CHANNELS, 
                                  len_output=LEN_OUTPUT, 
                                  filter_width = 2,
                                  speech_type='clean',
                                  bias = True)
        
    _, skips_noisy_batch = _wavnet(inputs=inputs_noisy,
                                  num_blocks=NUM_BLOCKS_NOISY, 
                                  num_layers=NUM_LAYERS_NOISY, 
                                  num_residual_channels=NUM_RESIDUAL_CHANNELS_NOISY, 
                                  num_skip_channels=NUM_SKIP_CHANNELS, 
                                  len_output=LEN_OUTPUT, 
                                  filter_width = 3,
                                  speech_type='noisy',
                                  bias = True)
                  
    skips_ll_batch = skips_clean_batch + skips_noisy_batch
    
    outputs_ll_batch = _post_processing(skips_ll_batch, 
                                        NUM_POST_LAYERS, 
                                        NUM_CLASSES, 
                                        'likli/')
    
    #prior speech model
    _, skips_prior_batch = _wavnet(inputs=inputs_clean,
                                   num_blocks=NUM_BLOCKS_CLEAN, 
                                   num_layers=NUM_LAYERS_CLEAN, 
                                   num_residual_channels=NUM_RESIDUAL_CHANNELS_CLEAN, 
                                   num_skip_channels=NUM_SKIP_CHANNELS, 
                                   len_output=LEN_OUTPUT, 
                                   filter_width = 2,
                                   speech_type='prior',
                                   bias = True,
                                   trainable=False)
    
    outputs_pr_batch = _post_processing(skips_prior_batch, 
                                        NUM_POST_LAYERS, 
                                        NUM_CLASSES, 
                                        'prior/', 
                                        trainable=False)
    
    #parameters of the speech model is not updated
    outputs_loglik_batch = tf.stop_gradient(outputs_pr_batch) + outputs_ll_batch
       
    return outputs_loglik_batch



def model_simple(inputs_noisy):
    """Build the BaWN simple model.
    Args:
    Returns:
      Logits.
    """
    
    _, skips_noisy_batch = _wavnet(inputs=inputs_noisy,
                                  num_blocks=NUM_BLOCKS_NOISY, 
                                  num_layers=NUM_LAYERS_NOISY, 
                                  num_residual_channels=NUM_RESIDUAL_CHANNELS_NOISY, 
                                  num_skip_channels=NUM_SKIP_CHANNELS, 
                                  len_output=LEN_OUTPUT, 
                                  filter_width = 3,
                                  speech_type='noisy',
                                  bias = True)
    
    outputs_sp_batch = _post_processing(skips_noisy_batch, 
                                        NUM_POST_LAYERS, 
                                        NUM_CLASSES, 
                                        'noisy/')
           
    return outputs_sp_batch



def loss(logits, labels):
    """Compute total loss
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]
    Returns:
      Loss tensor of type float.
    """
    #print(logits.shape)
    #print(labels.shape)
    # Calculate the average cross entropy loss across the batch.
    #labels = tf.cast(labels, tf.int32)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
  
    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')
    

    
def _step_filter(inputs, 
                 state,
                 future,
                 width, 
                 bias=True,
                 activation=None, 
                 name=None):
    """
    Performs efficient one-step filtering

    Inputs:
    intputs: the inputs tensor of size (batch_size, input_channels) at the specific time stamp
    currently supporting only 1
    state: the recurrent state tensor of size (batch_size, state_channels)
    bias: true if bias is included
    width: width of the convolution filters, currently supporting only 1 or 2 or 3
    name: name of the suffix of the variable
    activation: activation function applied

    Returns:
    output: the output of the convolution at that particular time
    
    """
    
    w = tf.get_variable(name+'/kernel')
    if width == 2:
        w_r = w[0, :, :]     #weight for recurrent state
        w_e = w[1, :, :]     #weight for current state
        output = tf.matmul(inputs, w_e) + tf.matmul(state, w_r)
    elif width == 3:
        w_r = w[0, :, :]     #weight for recurrent state
        w_e = w[1, :, :]     #weight for current state
        w_f = w[2, :, :]     #weight for future state
        output = tf.matmul(inputs, w_e) + tf.matmul(state, w_r) + tf.matmul(future, w_f)
    else:
        w = w[0, :, :]
        output = tf.matmul(inputs, w)
        
    if bias:
        b = tf.get_variable(name+'/bias')         #caution!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        output = tf.add(output, b)
    if activation:
        output = activation(output)
        
    return output



def _embed_filter(inputs, 
                  state,
                  future,
                  width, 
                  bias=True,
                  activation=None, 
                  name=None):
    """Compute one-step initial 2x1 conv using embedding
    Args:
      inputs: int32 ids returned from inputs
    Returns:
      output.
    """
    
    w = tf.get_variable(name+'/kernel')
    if width == 2:
        w_r = w[0, :, :]     #weight for recurrent state
        w_e = w[1, :, :]     #weight for current state
        output = tf.nn.embedding_lookup(w_e, inputs) \
               + tf.nn.embedding_lookup(w_r, state)
    elif width == 3:
        w_r = w[0, :, :]     #weight for recurrent state
        w_e = w[1, :, :]     #weight for current state
        w_f = w[2, :, :]     #weight for future state
        output = tf.nn.embedding_lookup(w_e, inputs) \
               + tf.nn.embedding_lookup(w_r, state) \
               + tf.nn.embedding_lookup(w_f, future)
    else:
        w = w[0, :, :]
        output = tf.nn.embedding_lookup(w, inputs)
        
    if bias:
        b = tf.get_variable(name+'/bias')         #caution!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        output = tf.add(output, b)
    if activation:
        output = activation(output)
        
    return output



def dilated_generation(inputs, state, future=None, width=2, bias=True, top=False, name=None):
    """
    Perform one sigle forward pass for one dilated conv layer 

                               |-> [gate]   -|                              |-> 1x1 conv -> skip output (size = skip_channel)
                               |             |-> (*)(size = gated_channel) -|
    input (size = in_channel) -|-> [filter] -|                              |-> 1x1 conv -|
                               |                                                          |-> (+) -> dense output (size = out_channel)
                               |----------------------------------------------------------|
    """
    assert name
    with tf.variable_scope(name, reuse=True) as scope:
                
        #generate gated output
        output_together = _step_filter(inputs, state, future, width, bias=False, name='together')
        _, together_channels = output_together.get_shape().as_list()
        residual_channels = int(together_channels/2)
                        
        #slice 2r channels into two r channels
        output_filter = tf.slice(output_together, [0,0], [-1,residual_channels], name='filter_part')
        output_gate = tf.slice(output_together, [0,residual_channels], [-1,-1], name='gate_part')
        #add activations
        output_filter = tf.tanh(output_filter, name='filter')
        output_gate = tf.sigmoid(output_gate, name='gate')
        output_gated = tf.multiply(output_filter, output_gate, name='filter_X_gate')
        
        #output_dense = residual + input
        if not top:
            output_residual = _step_filter(output_gated, None, None, 1, name='residual')
            output_dense = tf.add(inputs, output_residual)
        else:
            output_dense = None
        
        output_skip = _step_filter(output_gated, None, None, 1, name='skip')
        
    return (output_dense, output_skip, output_gated)


def post_processing_generation(inputs_agg, num_layers, name):
    """ 
    Performs post-processing (fully connected layers, 1 X 1 convolutions) for efficient generation
    
    inputs: sum of list of skip outputs of each dialted layer
    num_layers: number of layers, including the final output
    num_classes: the dimension of the final output 
    
    """
    #inputs_agg = tf.add_n(inputs, name='sum_skips')
    h = tf.nn.relu(inputs_agg)
    
    for l in range(num_layers-1):
        with tf.variable_scope('{}post_l{}'.format(name, l), reuse=True):
            h = _step_filter(h, None, None, 1, name='conv1d', activation=tf.nn.relu)
            
    #the last layer has no activation function
    with tf.variable_scope('{}post_l{}'.format(name, num_layers-1), reuse=True):
        outputs = _step_filter(h, None, None, 1, name='conv1d')
        
    return outputs


def model_history(inputs_clean, inputs_noisy):
    
    hs_clean_batch, _ = _wavnet(inputs=inputs_clean,
                                num_blocks=NUM_BLOCKS_CLEAN, 
                                num_layers=NUM_LAYERS_CLEAN, 
                                num_residual_channels=NUM_RESIDUAL_CHANNELS_CLEAN, 
                                num_skip_channels=NUM_SKIP_CHANNELS, 
                                len_output=1, 
                                filter_width = 2,
                                speech_type='clean',
                                bias = True)
    
    _, skips_noisy_batch = _wavnet(inputs=inputs_noisy,
                                   num_blocks=NUM_BLOCKS_NOISY, 
                                   num_layers=NUM_LAYERS_NOISY, 
                                   num_residual_channels=NUM_RESIDUAL_CHANNELS_NOISY, 
                                   num_skip_channels=NUM_SKIP_CHANNELS, 
                                   len_output=LEN_OUTPUT, 
                                   filter_width = 3,
                                   speech_type='noisy',
                                   bias = True)
    
    hs_prior_batch, _ = _wavnet(inputs=inputs_clean,
                                num_blocks=NUM_BLOCKS_CLEAN, 
                                num_layers=NUM_LAYERS_CLEAN, 
                                num_residual_channels=NUM_RESIDUAL_CHANNELS_CLEAN, 
                                num_skip_channels=NUM_SKIP_CHANNELS, 
                                len_output=1, 
                                filter_width = 2,
                                speech_type='prior',
                                bias = True,
                                trainable=False)
    
    return (hs_prior_batch, hs_clean_batch, skips_noisy_batch)
    
    
    
def _causal_generate(self,
                     inputs, 
                     num_blocks, 
                     num_layers, 
                     num_residual_channels,
                     batch_size, 
                     model_name=None):    
    
    h = inputs
    init_ops = []
    dequ_ops = []
    push_ops = []
    skips = []
    
    #for initial 2x1 conv layer only
    q = tf.FIFOQueue(1, dtypes=tf.int32, shapes=(batch_size, 1))
    dequ = q.dequeue()
    init = q.enqueue_many(tf.zeros((1, batch_size, 1), tf.int32))
    state_ = q.dequeue()
    push = q.enqueue([h])
    init_ops.append(init)
    push_ops.append(push)
    dequ_ops.append(dequ)
    
    with tf.variable_scope('', reuse=True) as scope:
        name = '{}b0-l0/pre'.format(model_name)
        h = bawn._embed_filter(inputs[:,0], state_[:,0], None, width=2, activation=tf.tanh, name=name)
                
    state_size = num_residual_channels
    
    for b in xrange(num_blocks):
        for i in xrange(num_layers):
            rate = 2 ** i
            name = '{}b{}-l{}'.format(model_name, b, i)
            
            top = bawn.check_boundries(num_blocks, num_layers, b, i)[1]
                            
            #make a length [rate] queue for each layer
            q = tf.FIFOQueue(rate, dtypes=tf.float32, shapes=(batch_size, state_size))
            dequ = q.dequeue_many(rate)
            init = q.enqueue_many(tf.zeros((rate, batch_size, state_size)))
            state_ = q.dequeue()
            push = q.enqueue([h]) #?
            #list of operations
            init_ops.append(init)
            push_ops.append(push)
            dequ_ops.append(dequ)
            
            h, skip = bawn.dilated_generation(h, state_, None, width=2, top=top, name=name)[0:2]
            skips.append(skip)
    
    skips_sum = tf.add_n(skips)
                            
    return (dequ_ops, init_ops, push_ops, skips_sum)    



def step_generation(inputs_clean,
                    skips_noisy,
                    batch_size):
    
    dequ_ops_prior, init_ops_prior, push_ops_prior, skips_prior = \
    _causal_generate(inputs_clean, 
                     NUM_BLOCKS_CLEAN, 
                     NUM_LAYERS_CLEAN, 
                     NUM_RESIDUAL_CHANNELS_CLEAN, 
                     batch_size=batch_size, 
                     model_name='prior/')
    
    dequ_ops_clean, init_ops_clean, push_ops_clean, skips_clean = \
    _causal_generate(inputs_clean, 
                     NUM_BLOCKS_CLEAN, 
                     NUM_LAYERS_CLEAN, 
                     NUM_RESIDUAL_CHANNELS_CLEAN,
                     batch_size=batch_size,
                     model_name='clean/')
    
    init_ops = init_ops_prior + init_ops_clean  #concatnate
    push_ops = push_ops_prior + push_ops_clean
    dequ_ops = dequ_ops_prior + dequ_ops_clean
    
    skips_likli = skips_clean + skips_noisy
    outputs_pr = post_processing_generation(skips_prior, NUM_POST_LAYERS, 'prior/')
    outputs_ll = post_processing_generation(skips_likli, NUM_POST_LAYERS, 'likli/')
    output_loglik = tf.add(outputs_pr, outputs_ll, name='output_loglik')      
    output_softmax = tf.nn.softmax(output_loglik)
    
    # compute mean 
    mean = tf.matmul(output_softmax, bins)
    
    out_ops = [mean]
    out_ops.extend(push_ops)
    
    return out_ops



def preprocess_inputs(inputs_clean, inputs_noisy):
    
    _, bins_center = mu_law_bins_tf(NUM_CLASSES)
    
    with tf.device('/gpu:0'):
        hs_prior_batch, _ = _wavnet(inputs=inputs_clean,
                                    num_blocks=NUM_BLOCKS_CLEAN, 
                                    num_layers=NUM_LAYERS_CLEAN, 
                                    num_residual_channels=NUM_RESIDUAL_CHANNELS_CLEAN, 
                                    num_skip_channels=NUM_SKIP_CHANNELS, 
                                    len_output=1, 
                                    filter_width = 2,
                                    speech_type='prior',
                                    bias = True,
                                    trainable=False)
        
        hs_clean_batch, _ = _wavnet(inputs=inputs_clean,
                                    num_blocks=NUM_BLOCKS_CLEAN, 
                                    num_layers=NUM_LAYERS_CLEAN, 
                                    num_residual_channels=NUM_RESIDUAL_CHANNELS_CLEAN, 
                                    num_skip_channels=NUM_SKIP_CHANNELS, 
                                    len_output=1, 
                                    filter_width = 2,
                                    speech_type='clean',
                                    bias = True)
    
        _, skips_noisy_batch = _wavnet(inputs=inputs_noisy,
                                       num_blocks=NUM_BLOCKS_NOISY, 
                                       num_layers=NUM_LAYERS_NOISY, 
                                       num_residual_channels=NUM_RESIDUAL_CHANNELS_NOISY, 
                                       num_skip_channels=NUM_SKIP_CHANNELS, 
                                       len_output=LEN_OUTPUT, 
                                       filter_width = 3,
                                       speech_type='noisy',
                                       bias = True)
        
    dequ_ops_prior, init_ops_prior, push_ops_prior, skips_prior = \
    _causal_generate(inputs_clean, 
                     NUM_BLOCKS_CLEAN, 
                     NUM_LAYERS_CLEAN, 
                     NUM_RESIDUAL_CHANNELS_CLEAN, 
                     batch_size=batch_size, 
                     model_name='prior/')
    
    dequ_ops_clean, init_ops_clean, push_ops_clean, skips_clean = \
    _causal_generate(inputs_clean, 
                     NUM_BLOCKS_CLEAN, 
                     NUM_LAYERS_CLEAN, 
                     NUM_RESIDUAL_CHANNELS_CLEAN,
                     batch_size=batch_size,
                     model_name='clean/')
    
    init_ops = init_ops_prior + init_ops_clean  #concatnate
    push_ops = push_ops_prior + push_ops_clean
    dequ_ops = dequ_ops_prior + dequ_ops_clean
    
    skips_likli = skips_clean + skips_noisy
    outputs_pr = post_processing_generation(skips_prior, NUM_POST_LAYERS, 'prior/')
    outputs_ll = post_processing_generation(skips_likli, NUM_POST_LAYERS, 'likli/')
    output_loglik = tf.add(outputs_pr, outputs_ll, name='output_loglik')      
    output_softmax = tf.nn.softmax(output_loglik)
    
    # compute mean 
    mean = tf.matmul(output_softmax, bins_center)
    
    out_ops = [mean]
    out_ops.extend(push_ops)
    
    return (init_ops, dequ_ops, out_ops)


def run_semi_online_v2(sess,
                       out_ops,
                       skips_noisy_batch, 
                       indices, 
                       inputs_noisy, 
                       num_samples):
    skips_noisy_sum = sess.run(skips_noisy_batch)
    predictions_ = []
    for step in xrange(num_samples):
        feed_dict = feed_dict={self.inputs_clean: indices,
                               self.skips_noisy: skips_noisy_sum[:,:,step]}
        output_dist = sess.run([out_ops], feed_dict=feed_dict)[0]
        #output dim = 1 x 256, it is 2D but we need 1D input to argmax
        indices = random_bins(NUM_CLASSES, output_dist)
        inputs = self.bins[indices]
        #inputs = np.array(np.matmul(output_dist,self.bins), dtype=np.float32)[:,None]
        #indices = np.digitize(inputs[:,0], self.bins, right=False)[:,None]
        predictions_.append(inputs)

        
