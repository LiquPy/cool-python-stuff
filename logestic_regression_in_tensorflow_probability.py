# with revisions
# based on https://medium.com/tensorflow/an-introduction-to-probabilistic-programming-now-available-in-tensorflow-probability-6dcc003ca29e?linkId=60908456

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import matplotlib.pyplot as plt

temperature_ = np.array([53, 57, 58, 64, 66, 67, 68, 69, 70, 70, 72, 73, 75, 75, 76, 78, 79, 81])
damage_ = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])

T = tf.convert_to_tensor(temperature_, dtype=tf.float32)
D = tf.convert_to_tensor(damage_, dtype=tf.float32)

beta = tfd.Normal(name='beta', loc=.3, scale=1000.).sample()
alpha = tfd.Normal(name='alpha', loc=-15., scale=1000.).sample()

p_deterministic = tfd.Deterministic(name='p', loc=1.0/(1. + tf.exp(beta*temperature_ + alpha))).sample()

def evaluate(tensors):
    #allows us to evaluate tensors whether we are operating in TF graph or eager mode
    if tf.executing_eagerly():
         return tf.contrib.framework.nest.pack_sequence_as(
             tensors,
             [t.numpy() if tf.contrib.framework.is_tensor(t) else t
             for t in tf.contrib.framework.nest.flatten(tensors)])
    with tf.Session() as sess:
        return sess.run(tensors)

def challenger_joint_log_prob(D, temperature_, alpha, beta):
    """
    Joint log probability optimization function.
        
    Args:
      D: The Data from the challenger disaster representing presence or 
         absence of defect
      temperature_: The Data from the challenger disaster, specifically the temperature on 
         the days of the observation of the presence or absence of a defect
      alpha: one of the inputs of the HMC
      beta: one of the inputs of the HMC
    Returns: 
      Joint log probability optimization function.
    """
    rv_alpha = tfd.Normal(loc=0., scale=1000.)
    rv_beta = tfd.Normal(loc=0., scale=1000.)
    logistic_p = 1.0/(1. + tf.exp(beta * tf.to_float(temperature_) + alpha))
    rv_observed = tfd.Bernoulli(probs=logistic_p)
  
    return (
        rv_alpha.log_prob(alpha)
        + rv_beta.log_prob(beta)
        + tf.reduce_sum(rv_observed.log_prob(D))
    )

[
    prior_alpha_,
    prior_beta_,
    p_deterministic_,
    D_
] = evaluate([
    alpha,
    beta,
    p_deterministic,
    D,
])

plt.scatter(temperature_, damage_)

plt.show()
