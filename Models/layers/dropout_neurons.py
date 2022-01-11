import tensorflow as tf

class DropoutNeurons(tf.keras.layers.Layer):
    """dropout neuron for co-smoothing"""

    def __init__(self,
                 rate: float = 0.5,
                 ):
        super(DropoutNeurons, self).__init__(name="DropoutNeurons")

        self.permute = tf.keras.layers.Permute((2,1))
        self.dropout = tf.keras.layers.SpatialDropout1D(rate)

    @tf.function
    def call(self, x, training: bool = False):
        # [batch_size, time, neurons]
        x = self.permute(x)
        x = self.dropout(x, training=training)
        # [batch_size, neurons, time]
        x = self.permute(x)
        # [batch_size, time, neurons]
        return x


