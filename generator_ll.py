#import matplotlib.pyplot as plt
from time import time
from six.moves import xrange
import numpy as np
import tensorflow as tf
import bawn
from utils import mu_law_bins, random_bins


class Model_Btch(object):

    def __init__(self,
                 len_input_clean=20477,
                 len_input_noisy=24570,
                 len_output=16384,
                 num_blocks_clean=4, 
                 num_layers_clean=10,
                 num_blocks_noisy=4, 
                 num_layers_noisy=10,
                 num_classes=256, 
                 num_post_layers=2,        #number of dense layers
                 num_residual_channels_clean=64,           #number of channels between hidden layers
                 num_residual_channels_noisy=64,           
                 num_skip_channels=256):
        
                
        self.len_output = len_output       
        self.num_classes = num_classes
                
        self.num_blocks_clean = num_blocks_clean
        self.num_layers_clean = num_layers_clean
        
        self.num_blocks_noisy = num_blocks_noisy
        self.num_layers_noisy = num_layers_noisy
        
        self.num_post_layers = num_post_layers
        
        self.num_residual_channels_clean = num_residual_channels_clean
        self.num_residual_channels_noisy = num_residual_channels_noisy
        
        self.num_skip_channels = num_skip_channels
        
        
        inputs_clean = tf.placeholder(tf.int32, shape=(None, len_input_clean))
        inputs_noisy = tf.placeholder(tf.float32, shape=(None, len_input_noisy))
        
        self.inputs_clean = inputs_clean
        self.inputs_noisy = inputs_noisy
        
        _, skips_prior_batch = bawn._wavnet(inputs=inputs_clean,
                                            num_blocks=num_blocks_clean, 
                                            num_layers=num_layers_clean, 
                                            num_residual_channels=num_residual_channels_clean, 
                                            num_skip_channels=num_skip_channels, 
                                            len_output=len_output, 
                                            filter_width=2,
                                            speech_type='prior',
                                            bias=True,
                                            trainable=False)
        
        outputs_pr_batch = bawn._post_processing(skips_prior_batch, 
                                                 num_post_layers, 
                                                 num_classes, 
                                                 'prior/', 
                                                 trainable=False)
        
        #noise model                
        _, skips_clean_batch = bawn._wavnet(inputs=inputs_clean,
                                            num_blocks=num_blocks_clean, 
                                            num_layers=num_layers_clean, 
                                            num_residual_channels=num_residual_channels_clean, 
                                            num_skip_channels=num_skip_channels, 
                                            len_output=len_output, 
                                            filter_width=2,
                                            speech_type='clean',
                                            bias=True,
                                            trainable=False)
        
        _, skips_noisy_batch = bawn._wavnet(inputs=inputs_noisy,
                                            num_blocks=num_blocks_noisy, 
                                            num_layers=num_layers_noisy, 
                                            num_residual_channels=num_residual_channels_noisy, 
                                            num_skip_channels=num_skip_channels, 
                                            len_output=len_output, 
                                            filter_width=3,
                                            speech_type='noisy',
                                            bias=True,
                                            trainable=False)
                      
        skips_ll_batch = skips_clean_batch + skips_noisy_batch
                
        outputs_ll_batch = bawn._post_processing(skips_ll_batch, 
                                                 num_post_layers, 
                                                 num_classes, 
                                                 'likli/',
                                                 trainable=False)
        
        outputs_loglik_batch = outputs_pr_batch + outputs_ll_batch
                
        
        self.outputs_softmax_batch = tf.nn.softmax(outputs_loglik_batch, dim=1)
        self.skips_noisy_batch = tf.add_n(skips_noisy_batch)
        self.skips_ll_batch = skips_ll_batch
        self.skips_prior_batch = skips_prior_batch
        self.outputs_ll_batch = outputs_ll_batch
        self.outputs_pr_batch = outputs_pr_batch
        self.outputs_loglik_batch = outputs_loglik_batch
        
        #params of batch training model
        self.saver = tf.train.Saver(tf.global_variables())
        
        
        # Create an ExponentialMovingAverage object
        ema = tf.train.ExponentialMovingAverage(decay=0.9999)
        real_var = [v for v in tf.global_variables()]
        shadow_name = []
        for v in tf.global_variables():
            if 'prior' in v.name:
                shadow_name.append(v.op.name)
            else:
                shadow_name.append(ema.average_name(v))
                
               
        self.saver_shadow = tf.train.Saver(dict(zip(shadow_name, real_var)))
        
  
    
    
class Generator(object):

    def __init__(self, batch_size=1, input_size=1):
        
        _, self.bins = mu_law_bins_tf(bawn.NUM_CLASSES)
        inputs_clean = tf.placeholder(tf.int32, [batch_size, input_size], name='inputs_clean')
        inputs_noisy = tf.placeholder(tf.float32, [batch_size, input_size], name='inputs_noisy')
        print 'Make Generator.'
        
        dequ_ops_prior, init_ops_prior, push_ops_prior, skips_prior = \
        self._causal_generate(inputs_clean, 
                              4, 10, 64, 
                              batch_size=batch_size, 
                              model_name='prior/')
        
        dequ_ops_clean, init_ops_clean, push_ops_clean, skips_clean = \
        self._causal_generate(inputs_clean, 
                              4, 10, 64,
                              batch_size=batch_size,
                              model_name='clean/')
        
        init_ops = init_ops_prior + init_ops_clean  #concatnate
        push_ops = push_ops_prior + push_ops_clean
        dequ_ops = dequ_ops_prior + dequ_ops_clean
        
        skips_noisy = tf.placeholder(skips_clean.dtype, skips_clean.shape, name='skips_noisy')
        skips_likli = skips_clean + skips_noisy
        outputs_pr = bawn.post_processing_generation(skips_prior, bawn.NUM_POST_LAYERS, 'prior/')
        outputs_ll = bawn.post_processing_generation(skips_likli, bawn.NUM_POST_LAYERS, 'likli/')
        output_loglik = tf.add(outputs_pr, outputs_ll, name='output_loglik')      #loglik for debug only
        output_softmax = tf.nn.softmax(output_loglik)
        
                
        #for flush states 
        out_ops_clean = [skips_clean]
        out_ops_clean.extend(push_ops_clean)
        out_ops_prior = [skips_prior]
        out_ops_prior.extend(push_ops_prior)
        out_ops = [output_softmax]
        out_ops.extend(push_ops)
        
        self.out_ops_clean = out_ops_clean
        self.out_ops_prior = out_ops_prior
        self.out_ops = out_ops
        self.inputs_clean = inputs_clean
        self.inputs_noisy = inputs_noisy
        self.init_ops = init_ops
        self.dequ_ops = dequ_ops
        self.skips_noisy = skips_noisy
        
        # for debug
        out_ops_skips_likli = [outputs_pr]
        out_ops_skips_likli.extend(push_ops_prior)
        self.out_ops_skips_likli = out_ops_skips_likli
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement=True
        config.log_device_placement=True
        config.intra_op_parallelism_threads=16
        config.inter_op_parallelism_threads=4
        self.sess = tf.Session(config=config)
        
        self.sess.run(self.init_ops)
        
        
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
            #h = tf.one_hot(inputs[:,0], 256, axis=1, dtype=tf.float32)
            #state_ = tf.one_hot(state_[:,0], 256, axis=1, dtype=tf.float32)
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
                
                       
        
    def run_offline(self, model, inputs_clean, inputs_noisy):
        feed_dict = {model.inputs_clean: inputs_clean,
                     model.inputs_noisy: inputs_noisy}
        output_dist = self.sess.run(model.outputs_softmax_batch, feed_dict=feed_dict)
        
        indices = np.argmax(output_dist, axis=1)
        
        predictions = np.array(self.bins[indices])
        #print predictions.shape
        plt.plot(predictions[0,:], label='pred')
        plt.legend()
        plt.xlabel('samples')
        plt.ylabel('signal')
        plt.show()
        return predictions
    
    
    def run_semi_online_real(self, model, inputs_clean, inputs_noisy, num_samples):
        skips_noisy_sum = self.sess.run(model.skips_noisy_batch, 
                                        feed_dict={model.inputs_noisy: inputs_noisy})
        predictions_ = []
        for step in xrange(num_samples):
            inputs_clean_ = inputs_clean[:, step:step+1]
            feed_dict = feed_dict={self.inputs_clean: inputs_clean_,
                                   self.skips_noisy: skips_noisy_sum[:,:,step]}
            output_dist = self.sess.run(self.out_ops, feed_dict=feed_dict)[0]
            inputs = np.array(self.bins[np.argmax(output_dist,axis=1)], dtype=np.float32)[:,None]
            predictions_.append(inputs)
            
            if step % 1000 == 0 and step != 0:
                predictions = np.concatenate(predictions_, axis=1)
                plt.plot(predictions[0,:], label='pred')
                plt.legend()
                plt.xlabel('samples from start')
                plt.ylabel('signal')
                plt.show()

        predictions = np.concatenate(predictions_, axis=1)
        return predictions
    
    
    def run_semi_online_real_v2(self, model, inputs_clean, inputs_noisy, num_samples):
        skips_noisy_sum = self.sess.run(model.skips_noisy_batch, 
                                        feed_dict={model.inputs_noisy: inputs_noisy})
        #skips_noisy_sum = np.swapaxes(skips_noisy_sum, 1, 2)
        test = []
        for step in xrange(num_samples):
            inputs_clean_ = inputs_clean[:, step:step+1]
            feed_dict = feed_dict={self.inputs_clean: inputs_clean_,
                                   self.skips_noisy: skips_noisy_sum[:,:,step]}
            skips_likli = self.sess.run(self.out_ops_skips_likli, feed_dict=feed_dict)[0]
            test.append(skips_likli)
            #print(skips_likli)
        
        return test
    
    
    def run_semi_online(self, model, hhhh, inputs_noisy, num_samples):
        skips_noisy_sum = self.sess.run(model.skips_noisy_batch, 
                                        feed_dict={model.inputs_noisy: inputs_noisy})
        predictions_ = []
        for step in xrange(num_samples):
            if step == 0:
                indices = hhhh[:, step:step+1]
            feed_dict = feed_dict={self.inputs_clean: indices,
                                   self.skips_noisy: skips_noisy_sum[:,:,step]}
            output_dist = self.sess.run([self.out_ops], feed_dict=feed_dict)[0]
       #    #output dim = 1 x 256, it is 2D but we need 1D input to argmax
       #    indices = random_bins(bawn.NUM_CLASSES, output_dist)
       #    inputs = self.bins[indices]
       #    #inputs = np.array(np.matmul(output_dist,self.bins), dtype=np.float32)[:,None]
       #    #indices = np.digitize(inputs[:,0], self.bins, right=False)[:,None]
       #    predictions_.append(inputs)
       #    
       #    if step % 1000 == 0 and step != 0:
       #        predictions = np.concatenate(predictions_, axis=1)
       #        plt.plot(predictions[0,:], label='pred')
       #        plt.legend()
       #        plt.xlabel('samples from start')
       #        plt.ylabel('signal')
       #        plt.show()

       #predictions = np.concatenate(predictions_, axis=1)
       #return predictions
        
        

        
class Generator_Hybrid(object):

    def __init__(self, len_pad=4093, batch_size=1, input_size=1):
        
        _, self.bins = mu_law_bins(bawn.NUM_CLASSES)
        self.len_pad = len_pad
        
        print 'Make Generator_Hybrid.'
        
        history_clean = tf.placeholder(tf.int32, [None, len_pad+1], name='history_clean')
        
        with tf.variable_scope("", reuse=True), tf.device('/gpu:0'):
            #clean part of the noise model                
            hs_clean_batch, _ = bawn._wavnet(inputs=history_clean,
                                             num_blocks=bawn.NUM_BLOCKS_CLEAN, 
                                             num_layers=bawn.NUM_LAYERS_CLEAN, 
                                             num_residual_channels=bawn.NUM_RESIDUAL_CHANNELS_CLEAN, 
                                             num_skip_channels=bawn.NUM_SKIP_CHANNELS, 
                                             len_output=1, 
                                             filter_width=2,
                                             speech_type='clean',
                                             bias=True,
                                             trainable=False)
                                
            #prior speech model
            hs_prior_batch, _ = bawn._wavnet(inputs=history_clean,
                                             num_blocks=bawn.NUM_BLOCKS_CLEAN, 
                                             num_layers=bawn.NUM_LAYERS_CLEAN, 
                                             num_residual_channels=bawn.NUM_RESIDUAL_CHANNELS_CLEAN, 
                                             num_skip_channels=bawn.NUM_SKIP_CHANNELS, 
                                             len_output=1, 
                                             filter_width=2,
                                             speech_type='prior',
                                             bias=True,
                                             trainable=False)
            
                
        inputs_clean = tf.placeholder(tf.int32, [batch_size, input_size], name='inputs_clean')
                        
        dequ_ops_prior, init_ops_prior, push_ops_prior, skips_prior = \
        self._causal_generate(inputs_clean,
                              hs_prior_batch,
                              4, 10, 64, 
                              batch_size=batch_size, 
                              model_name='prior/')
        
        dequ_ops_clean, init_ops_clean, push_ops_clean, skips_clean = \
        self._causal_generate(inputs_clean,
                              hs_clean_batch,
                              4, 10, 64,
                              batch_size=batch_size,
                              model_name='clean/')
        
        init_ops = init_ops_prior + init_ops_clean  #concatnate
        push_ops = push_ops_prior + push_ops_clean
        dequ_ops = dequ_ops_prior + dequ_ops_clean
        
        skips_noisy = tf.placeholder(skips_clean.dtype, skips_clean.shape, name='skips_noisy')
        skips_likli = skips_clean + skips_noisy
        outputs_pr = bawn.post_processing_generation(skips_prior, bawn.NUM_POST_LAYERS, 'prior/')
        outputs_ll = bawn.post_processing_generation(skips_likli, bawn.NUM_POST_LAYERS, 'likli/')
        output_loglik = tf.add(outputs_pr, outputs_ll, name='output_loglik')      #loglik for debug only
        output_softmax = tf.nn.softmax(output_loglik)
               
                
        #for flush states 
        out_ops_clean = [skips_clean]
        out_ops_clean.extend(push_ops_clean)
        out_ops_prior = [skips_prior]
        out_ops_prior.extend(push_ops_prior)
        out_ops = [output_softmax]
        out_ops.extend(push_ops)
        
        self.out_ops_clean = out_ops_clean
        self.out_ops_prior = out_ops_prior
        self.out_ops = out_ops
        self.inputs_clean = inputs_clean
        self.history_clean = history_clean
        self.init_ops = init_ops
        self.dequ_ops = dequ_ops
        self.skips_noisy = skips_noisy
        
        # for debug
        out_ops_skips_likli = [outputs_pr]
        out_ops_skips_likli.extend(push_ops_prior)
        self.out_ops_skips_likli = out_ops_skips_likli
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement=True
        config.log_device_placement=True
        config.intra_op_parallelism_threads=16
        config.inter_op_parallelism_threads=4
        self.sess = tf.Session(config=config)
        
        #self.sess.run(self.init_ops)
        
        
    def _causal_generate(self,
                         inputs,
                         hs_causal,
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
        init = q.enqueue([hs_causal[0]])
        state_ = q.dequeue()
        push = q.enqueue([h])
        init_ops.append(init)
        push_ops.append(push)
        dequ_ops.append(dequ)
        
        count = 1
        
        with tf.variable_scope('', reuse=True) as scope:
            #h = tf.one_hot(inputs[:,0], 256, axis=1, dtype=tf.float32)
            #state_ = tf.one_hot(state_[:,0], 256, axis=1, dtype=tf.float32)
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
                init = q.enqueue_many(tf.transpose(hs_causal[count][:,:,-rate-1:-1], perm=[2,0,1]))
                state_ = q.dequeue()
                push = q.enqueue([h]) #?
                #list of operations
                init_ops.append(init)
                push_ops.append(push)
                dequ_ops.append(dequ)
                
                h, skip = bawn.dilated_generation(h, state_, None, width=2, top=top, name=name)[0:2]
                skips.append(skip)
                count += 1
        
        skips_sum = tf.add_n(skips)
                                
        return (dequ_ops, init_ops, push_ops, skips_sum)  
    

    def run_semi_online_v2(self, model, inputs_clean, inputs_noisy, num_samples):
        dump = self.sess.run(self.init_ops, 
                             feed_dict={self.history_clean: inputs_clean[:,0:self.len_pad+1]})
        skips_noisy_sum = self.sess.run(model.skips_noisy_batch, 
                                        feed_dict={model.inputs_noisy: inputs_noisy})
        indices = inputs_clean[:,self.len_pad:self.len_pad+1]
        predictions_ = []
        for step in xrange(num_samples):
            indices = inputs_clean[:,self.len_pad+step:self.len_pad+1+step]
            feed_dict = feed_dict={self.inputs_clean: indices,
                                   self.skips_noisy: skips_noisy_sum[:,:,step]}
            output_dist = self.sess.run(self.out_ops, feed_dict=feed_dict)[0]
            indices = np.argmax(output_dist, axis=1)[:,None]
            inputs = self.bins[indices[:,0]]
            #inputs = np.array(np.matmul(output_dist,self.bins), dtype=np.float32)[:,None]
            #indices = np.digitize(inputs[:,0], self.bins, right=False)[:,None]
            predictions_.append(inputs[:,None])
            
            if step % 1000 == 0 and step != 0:
                predictions = np.concatenate(predictions_, axis=1)
                plt.plot(predictions[0,:], label='pred')
                plt.legend()
                plt.xlabel('samples from start')
                plt.ylabel('signal')
                plt.show()

        predictions = np.concatenate(predictions_, axis=1)
        dump = self.sess.run(self.dequ_ops)
        return predictions
    
    
    def run_offline(self, model, inputs_clean, inputs_noisy):
        feed_dict = {model.inputs_clean: inputs_clean,
                     model.inputs_noisy: inputs_noisy}
        output_dist = self.sess.run(model.outputs_softmax_batch, feed_dict=feed_dict)
        
        indices = np.argmax(output_dist, axis=1)
        
        predictions = np.array(self.bins[indices])
        #print predictions.shape
        #plt.plot(predictions[0,:], label='pred')
        #plt.legend()
        #plt.xlabel('samples')
        #plt.ylabel('signal')
        #plt.show()
        return predictions
    
    
    
    
class Generator_Hybrid_v2(object):

    def __init__(self, 
                 len_pad=4093, 
                 len_input_noisy=24570,
                 len_output=16384, 
                 batch_size=1, 
                 input_size=1):
        
        self.bins_edge, self.bins_center = mu_law_bins(bawn.NUM_CLASSES)
        self.len_pad = len_pad
        self.len_output = len_output
        
        print 'Make Generator_Hybrid_v2.'
        
        history_clean = tf.placeholder(tf.int32, [None, len_pad+1], name='history_clean')
        inputs_noisy = tf.placeholder(tf.float32, [None, len_input_noisy], name='inputs_noisy')
        
        with tf.variable_scope("", reuse=True), tf.device('/gpu:0'):
            #clean part of the noise model                
            hs_clean_batch, _ = bawn._wavnet(inputs=history_clean,
                                             num_blocks=bawn.NUM_BLOCKS_CLEAN, 
                                             num_layers=bawn.NUM_LAYERS_CLEAN, 
                                             num_residual_channels=bawn.NUM_RESIDUAL_CHANNELS_CLEAN, 
                                             num_skip_channels=bawn.NUM_SKIP_CHANNELS, 
                                             len_output=1, 
                                             filter_width=2,
                                             speech_type='clean',
                                             bias=True,
                                             trainable=False)
                                
            #prior speech model
            hs_prior_batch, _ = bawn._wavnet(inputs=history_clean,
                                             num_blocks=bawn.NUM_BLOCKS_CLEAN, 
                                             num_layers=bawn.NUM_LAYERS_CLEAN, 
                                             num_residual_channels=bawn.NUM_RESIDUAL_CHANNELS_CLEAN, 
                                             num_skip_channels=bawn.NUM_SKIP_CHANNELS, 
                                             len_output=1, 
                                             filter_width=2,
                                             speech_type='prior',
                                             bias=True,
                                             trainable=False)
            
            _, skips_noisy_batch = bawn._wavnet(inputs=inputs_noisy,
                                                num_blocks=bawn.NUM_BLOCKS_NOISY, 
                                                num_layers=bawn.NUM_LAYERS_NOISY, 
                                                num_residual_channels=bawn.NUM_RESIDUAL_CHANNELS_NOISY, 
                                                num_skip_channels=bawn.NUM_SKIP_CHANNELS, 
                                                len_output=len_output, 
                                                filter_width=3,
                                                speech_type='noisy',
                                                bias=True,
                                                trainable=False)
            
                
        inputs_clean = tf.placeholder(tf.int32, [batch_size, input_size], name='inputs_clean')
                        
        dequ_ops_prior, init_ops_prior, push_ops_prior, skips_prior = \
        self._causal_generate(inputs_clean,
                              hs_prior_batch,
                              4, 10, 64, 
                              batch_size=batch_size, 
                              model_name='prior/')
        
        dequ_ops_clean, init_ops_clean, push_ops_clean, skips_clean = \
        self._causal_generate(inputs_clean,
                              hs_clean_batch,
                              4, 10, 64,
                              batch_size=batch_size,
                              model_name='clean/')
        
        init_ops = init_ops_prior + init_ops_clean  #concatnate
        push_ops = push_ops_prior + push_ops_clean
        dequ_ops = dequ_ops_prior + dequ_ops_clean
        
        skips_noisy = tf.placeholder(skips_clean.dtype, skips_clean.shape, name='skips_noisy')
        skips_likli = skips_clean + skips_noisy
        outputs_pr = bawn.post_processing_generation(skips_prior, bawn.NUM_POST_LAYERS, 'prior/')
        outputs_ll = bawn.post_processing_generation(skips_likli, bawn.NUM_POST_LAYERS, 'likli/')
        output_loglik = tf.add(outputs_pr, outputs_ll, name='output_loglik')      #loglik for debug only
        output_softmax = tf.nn.softmax(output_loglik)
               
                
        #for flush states 
        out_ops_clean = [skips_clean]
        out_ops_clean.extend(push_ops_clean)
        out_ops_prior = [skips_prior]
        out_ops_prior.extend(push_ops_prior)
        out_ops = [output_softmax]
        out_ops.extend(push_ops)
        
        self.out_ops_clean = out_ops_clean
        self.out_ops_prior = out_ops_prior
        self.out_ops = out_ops
        self.inputs_clean = inputs_clean
        self.history_clean = history_clean
        self.inputs_noisy = inputs_noisy
        self.init_ops = init_ops
        self.dequ_ops = dequ_ops
        self.skips_noisy = skips_noisy
        self.skips_noisy_sum = tf.add_n(skips_noisy_batch)
               
               
        
    def _causal_generate(self,
                         inputs,
                         hs_causal,
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
        init = q.enqueue([hs_causal[0]])
        state_ = q.dequeue()
        push = q.enqueue([h])
        init_ops.append(init)
        push_ops.append(push)
        dequ_ops.append(dequ)
        
        count = 1
        
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
                init = q.enqueue_many(tf.transpose(hs_causal[count][:,:,-rate-1:-1], perm=[2,0,1]))
                state_ = q.dequeue()
                push = q.enqueue([h]) #?
                #list of operations
                init_ops.append(init)
                push_ops.append(push)
                dequ_ops.append(dequ)
                
                h, skip = bawn.dilated_generation(h, state_, None, width=2, top=top, name=name)[0:2]
                skips.append(skip)
                count += 1
        
        skips_sum = tf.add_n(skips)
                                
        return (dequ_ops, init_ops, push_ops, skips_sum)  
    

    def run_semi_online(self, sess, inputs_clean, inputs_noisy, num_samples):
        dump = sess.run(self.init_ops, 
                        feed_dict={self.history_clean: inputs_clean[:,0:self.len_pad+1]})
        skips_noisy_sum = sess.run(self.skips_noisy_sum, 
                                   feed_dict={self.inputs_noisy: inputs_noisy})
        indices = inputs_clean[:,self.len_pad:self.len_pad+1]
        predictions_ = []
        for step in xrange(num_samples):
            #indices = inputs_clean[:,self.len_pad+step:self.len_pad+1+step]
            feed_dict = feed_dict={self.inputs_clean: indices,
                                   self.skips_noisy: skips_noisy_sum[:,:,step]}
            output_dist = sess.run(self.out_ops, feed_dict=feed_dict)[0]
            #indices = np.argmax(output_dist, axis=1)[:,None]
            #inputs = self.bins_center[indices[:,0]].astype(np.float32)
            inputs = np.matmul(output_dist, self.bins_center).astype(np.float32)
            indices = np.digitize(inputs, self.bins_edge, right=False)[:,None]
            predictions_.append(indices)
            
        predictions = np.concatenate(predictions_, axis=1)
        dump = sess.run(self.dequ_ops)
        
        return predictions
    
    
    def run_offline(self, sess, model, inputs_clean, inputs_noisy):
        feed_dict = {model.inputs_clean: inputs_clean,
                     model.inputs_noisy: inputs_noisy}
        output_dist = sess.run(model.outputs_softmax_batch, feed_dict=feed_dict)
        
        indices = np.argmax(output_dist, axis=1)
        
        predictions = np.array(self.bins_center[indices])
        
        return predictions