from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import time
import pickle

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import bawn
from generator_ll import Generator_Hybrid_v2

LOG_DIR = '/scratch/logs/run22'
PRE_TRAINED = '/scratch/logs/run#/bawn_pr.ckpt-299131'
NUM_GPUS = 8
LOG_DEVICE_PLACEMENT = False


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999
INITIAL_LEARNING_RATE = 0.001
ANNEALING_RATE = 0.9886
MAX_STEPS = 3000000
NUM_STEPS_PER_DECAY = 1000
PERIOD_SUMMARY = 120
PERIOD_CHECKPOINT = 600


def tower_loss(scope, segments_clean, segments_noisy, labels):
    """Calculate the total loss on a single tower running the BAWN model.
    Args:
      scope: unique prefix string identifying the BAWN tower, e.g. 'tower_0'
    Returns:
       Tensor of shape [] containing the total loss for a batch of data
    """
    #segments = tf.Print(segments, [segments[0,4094:]], message=scope, summarize=10)
    #labels = tf.Print(labels, [labels[0,:-1]], message=scope, summarize=10)
    
    # Build inference Graph.
    logits = bawn.model_denoise(segments_clean, segments_noisy)
        
    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = bawn.loss(tf.transpose(logits, perm=[0, 2, 1]), labels)
  
    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)
  
    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')
    
    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        loss_name = re.sub('%s_[0-9]*/' % bawn.TOWER_NAME, '', l.op.name)
        tf.summary.scalar(loss_name, l)
    
    return total_loss


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)
      
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)
    
        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
    
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads



def train():
    """Train BAWN for a number of steps."""
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # training data initializers
        with tf.name_scope('input'):
            queue_raw = tf.train.range_input_producer(data_labels.shape[0])
            subdiv = queue_raw.dequeue_many(NUM_GPUS)
                         
            input_clean = tf.placeholder_with_default(tf.zeros([NUM_GPUS, data_clean.shape[1]], tf.int32), 
                                          shape=[NUM_GPUS, data_clean.shape[1]], name='cleans')
            input_noisy = tf.placeholder_with_default(tf.zeros([NUM_GPUS, data_noisy.shape[1]], tf.float32), 
                                          shape=[NUM_GPUS, data_noisy.shape[1]], name='noisys')
            input_labels = tf.placeholder_with_default(tf.zeros([NUM_GPUS, data_labels.shape[1]], tf.int32), 
                                          shape=[NUM_GPUS, data_labels.shape[1]], name='labels')
            
            queue_proc = tf.train.range_input_producer(NUM_GPUS, shuffle=False)
            index = queue_proc.dequeue_many(bawn.BATCH_SIZE)
            inputs_list = [input_clean, input_noisy, input_labels]

        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * num_gpus.
        global_step = tf.train.create_global_step()
        
        # Decay the learning rate based on the number of steps.
        lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step,
                                        NUM_STEPS_PER_DECAY, ANNEALING_RATE, staircase=True)
          
        # Create an optimizer that performs gradient descent.
        opt = tf.train.AdamOptimizer(lr)
        
        # Calculate the gradients for each model tower.
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(NUM_GPUS):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (bawn.TOWER_NAME, i)) as scope:
                        with tf.device('/cpu:0'):
                            cleans, noisys, labels = [tf.gather(t, index) for t in inputs_list]
                        # Calculate the loss for one tower of the BAWN model. This function
                        # constructs the entire BAWN model but shares the variables across
                        # all towers.
                        loss = tower_loss(scope, cleans, noisys, labels)
                        
                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()
            
                        # Retain the summaries from the final tower.
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
            
                        # Calculate the gradients for the batch of data on this BAWN tower.
                        grads = opt.compute_gradients(loss)
            
                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)
        
        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)
    
        # Add a summary to track the learning rate.
        summaries.append(tf.summary.scalar('learning_rate', lr))
        
        # build generator for "cleaned" inputs
        generator = Generator_Hybrid_v2(batch_size=NUM_GPUS)
    
        # Apply the gradients to adjust the shared variables and increment the global step.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        
        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
                            MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        
        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)
    
        # Create a saver.
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=4096)
        # Create a saver that restores only the pre-trained variables.
        pre_train_saver = tf.train.Saver([v for v in tf.global_variables() if 'prior' in v.name])
        # saver that restores moving averaged variables
        real_var = [v for v in tf.global_variables() if 'prior' in v.name]
        shadow_name = []
        for v in tf.global_variables():
            if 'prior' in v.name:
                shadow_name.append(variable_averages.average_name(v))
        pre_train_saver_shadow = tf.train.Saver(dict(zip(shadow_name, real_var)))        
      
        # Define an init function that loads the pretrained checkpoint.
        def load_pretrain(sess):
            pre_train_saver_shadow.restore(sess, PRE_TRAINED)
    
        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)
        
        sess_config = tf.ConfigProto(allow_soft_placement=True, 
                                     log_device_placement=LOG_DEVICE_PLACEMENT,
                                     intra_op_parallelism_threads=20,
                                     inter_op_parallelism_threads=4)                        
        
        # Superviser 
        sv = tf.train.Supervisor(logdir=LOG_DIR
                                 ,init_fn=load_pretrain
                                 ,summary_op=summary_op
                                 ,saver=saver
                                 ,save_model_secs=PERIOD_CHECKPOINT
                                 ,save_summaries_secs=PERIOD_SUMMARY
                                 ,checkpoint_basename='bawn_ll.ckpt')
                
        #sess = sv.prepare_or_wait_for_session(config=sess_config)
        
        with sv.managed_session(config=sess_config, start_standard_services=False) as sess:
                        
            print('Starting services and queue runners...')
            # start the queue runner after feed_dict so that the desired elements are enqueued
            sv.start_standard_services(sess)
            sv.start_queue_runners(sess)
                                     
            losses = []    
            start = sess.run(global_step)
            for step in xrange(start, MAX_STEPS):
                if sv.should_stop():
                    print('SB!!!!!!!!!')
                    break
                start_time = time.time()
                subpick = sess.run(subdiv)
                print('Generating using {}'.format(subpick))
                tic = time.time()
                predictions = generator.run_semi_online(sess, 
                                                        data_clean[subpick, :], 
                                                        data_noisy[subpick, :], 
                                                        bawn.LEN_OUTPUT)
                pred_clean = np.concatenate([data_clean[subpick, 0:bawn.LEN_PAD], predictions], axis=1)
                toc = time.time()
                print('Generating took {} seconds.'.format(toc-tic))
                
                feed_dict = {input_clean:pred_clean,  
                             input_noisy:data_noisy[subpick, :], 
                             input_labels:data_labels[subpick, :]}
                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
                
                duration = time.time() - start_time
            
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                                      
                if step % 1 == 0:
                    num_examples_per_step = bawn.BATCH_SIZE * NUM_GPUS
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration / NUM_GPUS
            
                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print (format_str % (datetime.now(), step, loss_value,
                                       examples_per_sec, sec_per_batch))
                    losses.append(loss_value)
                    pickle.dump(losses, open(os.path.join(LOG_DIR, 'losses.p'), "wb"))
      
            
if __name__ == '__main__':
    #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    #os.environ["CUDA_VISIBLE_DEVICES"]="0"
    with tf.device('/cpu:0'):
        data_clean, data_noisy, data_labels = bawn.load_data_likli('clean_train.mat', 'noisy_train.mat', 'target_train.mat')
        data_clean = data_clean[:32,:]
        data_noisy = data_noisy[:32,:]
        data_labels = data_labels[:32,:]
    train()