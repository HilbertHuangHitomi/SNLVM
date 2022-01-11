import tensorflow_probability as tfp
import tensorflow as tf

def vMF_kldiv_loss(compute_KLD=False):
    '''
    Input format: (g_?, mean_?, logvar_?)
    '''
    @tf.function
    def loss_fun(g_x):
        if compute_KLD:
            dist_posterior = tfp.distributions.VonMisesFisher(g_x[1], g_x[2])
            dist_prior = tfp.distributions.SphericalUniform(g_x[1].shape[1])
            return tf.reduce_sum(tfp.distributions.kl_divergence(
                dist_posterior, dist_prior, allow_nan_stats=False, name=None))
        else:
            #return tf.reduce_sum(g_x[2])
            return 0
    return loss_fun

