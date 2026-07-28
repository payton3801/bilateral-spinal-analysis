import tensorflow as tf
from tensorflow.keras.regularizers import Regularizer

class DynamicL2(Regularizer):
    """ An L2 regularizer with modifiable scale and ramping. """
    def __init__(self, scale):
        """Creates a new dynamically weighted regularizer.

        Parameters
        ----------
        scale : float
            A multiplier that increases or decreases the 
            penalty for a given parameter matrix.
        """
        self.scale = tf.Variable(scale, trainable=False)

    def __call__(self, x):
        """Computes the regularization penalty of the current weights.
        
        Parameters
        ----------
        x : tf.Variable
            The weights to be regularized.

        Returns
        -------
        tf.Tensor
            A scalar tensor representing the penalty paid for `x`.
        """
        return self.scale * 0.5 * tf.norm(x)**2

    def get_config(self):
        """Returns a serialized representation of the Regularizer.
        
        See the TensorFlow documentation for an explanation of serialization: 
        https://www.tensorflow.org/guide/keras/save_and_serialize#custom_objects
        
        Returns
        -------
        dict
            A dictionary that can be used to recreate this regularizer.
        """
        return {'scale': self.scale.numpy()}
    
    def update_config(self, config):
        """Updates the weight of the regularizer.
        
        Parameters
        ----------
        config : dict
            A dictionary containing the new scaling parameter.
        """
        self.scale.assign(config['scale'])

class DynamicL1(Regularizer):
    """ An L1 regularizer with modifiable scale and ramping. """
    def __init__(self, scale):
        """Creates a new dynamically weighted regularizer.

        Parameters
        ----------
        scale : float
            A multiplier that increases or decreases the 
            penalty for a given parameter matrix.
        """
        self.scale = tf.Variable(scale, trainable=False)

    def __call__(self, x):
        """Computes the regularization penalty of the current weights.
        
        Parameters
        ----------
        x : tf.Variable
            The weights to be regularized.

        Returns
        -------
        tf.Tensor
            A scalar tensor representing the penalty paid for `x`.
        """
        return self.scale * tf.reduce_sum(tf.math.abs(x))

    def get_config(self):
        """Returns a serialized representation of the Regularizer.
        
        See the TensorFlow documentation for an explanation of serialization: 
        https://www.tensorflow.org/guide/keras/save_and_serialize#custom_objects
        
        Returns
        -------
        dict
            A dictionary that can be used to recreate this regularizer.
        """
        return {'scale': self.scale.numpy()}
    
    def update_config(self, config):
        """Updates the weight of the regularizer.
        
        Parameters
        ----------
        config : dict
            A dictionary containing the new scaling parameter.
        """
        self.scale.assign(config['scale'])

class DynamicL1L2(Regularizer):
    """ An L1 regularizer with modifiable scale and ramping. """
    def __init__(self, l1_scale, l2_scale):
        """Creates a new dynamically weighted regularizer.

        Parameters
        ----------
        scale : float
            A multiplier that increases or decreases the 
            penalty for a given parameter matrix.
        """
        self.l1_scale = tf.Variable(l1_scale, trainable=False)
        self.l2_scale = tf.Variable(l2_scale, trainable=False)

    def __call__(self, x):
        """Computes the regularization penalty of the current weights.
        
        Parameters
        ----------
        x : tf.Variable
            The weights to be regularized.

        Returns
        -------
        tf.Tensor
            A scalar tensor representing the penalty paid for `x`.
        """
        l2_loss = self.l2_scale * 0.5 * tf.norm(x)**2
        l1_loss = self.l1_scale * tf.reduce_sum(tf.math.abs(x))
        
        return l1_loss + l2_loss

    def get_config(self):
        """Returns a serialized representation of the Regularizer.
        
        See the TensorFlow documentation for an explanation of serialization: 
        https://www.tensorflow.org/guide/keras/save_and_serialize#custom_objects
        
        Returns
        -------
        dict
            A dictionary that can be used to recreate this regularizer.
        """
        return {'l1_scale': self.l1_scale.numpy(), 'l2_scale': self.l2_scale.numpy()}
    
    def update_config(self, config):
        """Updates the weight of the regularizer.
        
        Parameters
        ----------
        config : dict
            A dictionary containing the new scaling parameter.
        """
        self.l1_scale.assign(config['l1_scale'])
        self.l2_scale.assign(config['l2_scale'])
        
