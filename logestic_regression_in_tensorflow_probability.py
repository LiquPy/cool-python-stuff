# with revisions
# based on https://medium.com/tensorflow/an-introduction-to-probabilistic-programming-now-available-in-tensorflow-probability-6dcc003ca29e?linkId=60908456

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp
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



# Hamiltonian Monte Carlo
number_of_steps = 60000
burnin = 50000

# Set the chain's start state.
initial_chain_state = [
    0. * tf.ones([], dtype=tf.float32, name="init_alpha"),
    0. * tf.ones([], dtype=tf.float32, name="init_beta")
]

# Since HMC operates over unconstrained space, we need to transform the
# samples so they live in real-space.
unconstraining_bijectors = [
    tfp.bijectors.Identity(),
    tfp.bijectors.Identity()
]

# Define a closure over our joint_log_prob.
unnormalized_posterior_log_prob = lambda *args: challenger_joint_log_prob(D, temperature_, *args)

# Initialize the step_size. (It will be automatically adapted.)
with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    step_size = tf.get_variable(
        name='step_size',
        initializer=tf.constant(0.5, dtype=tf.float32),
        trainable=False,
        use_resource=True
    )

# Defining the HMC
hmc=tfp.mcmc.TransformedTransitionKernel(
    inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=unnormalized_posterior_log_prob,
        num_leapfrog_steps=2,
        step_size=step_size,
        step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(),
        state_gradients_are_stopped=True),
    bijector=unconstraining_bijectors)

# Sampling from the chain.
[
    posterior_alpha,
    posterior_beta
], kernel_results = tfp.mcmc.sample_chain(
    num_results = number_of_steps,
    num_burnin_steps = burnin,
    current_state=initial_chain_state,
    kernel=hmc)

# Initialize any created variables for preconditions
init_g = tf.global_variables_initializer()

evaluate(init_g)
[
    posterior_alpha_,
    posterior_beta_,
    kernel_results_
] = evaluate([
    posterior_alpha,
    posterior_beta,
    kernel_results
])


print("acceptance rate: {}".format(
    kernel_results_.inner_results.is_accepted.mean()))
print("final step size: {}".format(
    kernel_results_.inner_results.extra.step_size_assign[-100:].mean()))

alpha_samples_ = posterior_alpha_[burnin::8]
beta_samples_ = posterior_beta_[burnin::8]

# Visualization
plt.scatter(temperature_, damage_)

plt.show()
