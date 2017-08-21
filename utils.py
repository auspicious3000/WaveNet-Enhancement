import numpy as np
import tensorflow as tf
from scipy.io import wavfile

def normalize(data):
    temp = np.float32(data) - np.average(data)
    out = temp / np.max(np.abs(temp))
    return out


def make_batch(path):
    rate, data = wavfile.read(path)
    #only use the 1st channel
    data = data[:, 0]
    data_ = normalize(data)
    bins, bins_center = mu_law_bins(256)
    inputs = np.digitize(data_[0:-1], bins, right=False)
    inputs = bins_center[inputs][None, :, None]
    #predict sample 1 to end using 0 to end-1
    targets = np.digitize(data_[1::], bins, right=False)[None, :]
    return (inputs, targets)


def make_batch_padded(path, num_layers = 14):
    rate, data = wavfile.read(path)
    #only use the 1st channel
    data = data[:, 0]
    data_ = normalize(data)
    bins, bins_center = mu_law_bins(256)
    inputs = np.digitize(data_[0:-1], bins, right=False)
    inputs = bins_center[inputs][None, :, None]
    #predict sample 1 to end using 0 to end-1
    targets = np.digitize(data_[1::], bins, right=False)[None, :]
    
    base = 2 ** num_layers
    _, width, _ = inputs.shape
    #crop the width to make it multiple of base
    width_cropped = int(np.floor(width * 1.0 / base) * base)
    inputs_padded = np.pad(inputs[:, 0:width_cropped, :], ((0, 0), (base - 1, 0), (0, 0)), 'constant')
    targets_padded = targets[:, 0:width_cropped]
    
    return (inputs_padded, targets_padded)


def mu_law_bins(num_bins):
    """ 
    this functions returns the mu-law bin (right) edges and bin centers, with num_bins number of bins 
    
    """
    #all edges
    bins_edge = np.linspace(-1, 1, num_bins + 1)
    #center of all edges
    bins_center = np.linspace(-1 + 1.0 / num_bins, 1 - 1.0 / num_bins, num_bins)
    #get the right edges
    bins_trunc = bins_edge[1:]
    #if sample >= right edges, it might be assigned to the next bin, add 0.1 to avoid this
    bins_trunc[-1] += 0.1
    #convert edges and centers to mu-law scale
    bins_edge_mu = np.multiply(np.sign(bins_trunc), (num_bins ** np.absolute(bins_trunc) - 1) / (num_bins - 1))
    bins_center_mu = np.multiply(np.sign(bins_center), (num_bins ** np.absolute(bins_center) - 1) / (num_bins - 1))
    
    return (bins_edge_mu, bins_center_mu)


def mu_law_bins_tf(num_bins):
    """ 
    this functions returns the mu-law bin (right) edges and bin centers, with num_bins number of bins 
    
    """
    #all edges
    bins_edge = tf.linspace(-1.0, 1.0, num_bins + 1)
    #center of all edges
    bins_center = tf.linspace(-1.0 + 1.0 / num_bins, 1.0 - 1.0 / num_bins, num_bins)
    #get the right edges
    bins_trunc = tf.concat([bins_edge[1:-1], [1.1]], 0)
    #if sample >= right edges, it might be assigned to the next bin, add 0.1 to avoid this
    #convert edges and centers to mu-law scale
    bins_edge_mu = tf.multiply(tf.sign(bins_trunc), (num_bins ** tf.abs(bins_trunc) - 1) / (num_bins - 1))
    bins_center_mu = tf.multiply(tf.sign(bins_center), (num_bins ** tf.abs(bins_center) - 1) / (num_bins - 1))
       
    return (bins_edge_mu, bins_center_mu)


def random_samples(bins, dist):
    """
    returns random samples from multiple distributions
    
    dist: N * 256 array
    samples : N * 1
    """
    N = dist.shape[0]
    samples = np.empty([N,1], dtype=np.float32)
    
    for i in range(N):
        smpl = np.random.choice(bins, p=dist[i,:]/np.sum(dist[i,:]))
        samples[i,0] = smpl.astype(np.float32)
        
    return samples


def random_bins(num_classes, dist):
    """
    returns random bins from multiple distributions
    
    dist: N * 256 array
    bins : N * 1
    """
    N = dist.shape[0]
    bins = np.empty([N,1], dtype=np.int32)
    
    for i in range(N):
        smpl = np.random.choice(num_classes, p=dist[i,:]/np.sum(dist[i,:]))
        bins[i,0] = smpl
        
    return bins    
