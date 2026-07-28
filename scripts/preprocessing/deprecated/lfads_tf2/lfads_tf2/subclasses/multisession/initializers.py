import tensorflow as tf

class PCRInitializer(tf.keras.initializers.Initializer):

    def __init__(self, matrix, matrix_type=None):
        self.matrix = matrix
        self.matrix_type = matrix_type

    def __call__(self, shape, dtype=None):
        # input matrix should be larger than shape, as defined by script 
        # that makes datasets
        if self.matrix_type == 'bias':
            # if dealing with a bias vector
            weights = self.matrix[:shape[0]]
        elif self.matrix_type == 'weight':
            # otherwise dealing with weights matrix
            weights = self.matrix[:shape[0],:shape[1]]
        return tf.convert_to_tensor(weights, dtype=dtype)

    def get_config(self):  # To support serialization
        return {'matrix': self.matrix, 'matrix_type': self.matrix_type}