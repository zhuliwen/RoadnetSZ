from keras import backend as K

class L2ReLURegularizer(object):
    """Regularizer for L2 + ReLU regularization.
    # Arguments
        l: Float; L2 regularization factor.
    """

    def __init__(self, l=0.):
        self.l = K.cast_to_floatx(l)

    def __call__(self, x):
        regularization = 0.
        if self.l:
            squared_relu_q_values = K.square(K.relu(x))
            batch_squared_relu_q_values = K.sum(squared_relu_q_values, axis=[1,2])
            mean_squared_relu_q_values = K.mean(batch_squared_relu_q_values)
            regularization += self.l * mean_squared_relu_q_values
        return regularization

    def get_config(self):
        return {'l': float(self.l)}
