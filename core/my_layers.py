import keras.backend as K
from keras.engine.topology import Layer
from keras.layers.convolutional import Convolution1D
from keras.utils.np_utils import conv_output_length
from keras.engine import InputSpec
import numpy as np
import sys


class Attention(Layer):
    """
    Attention layer for use with keras modules
    
    
    Supports 'attsum' and 'attmean'.
    Attsum is the more common method.
    Attmean sort of disrupts the balance of the numbers because of the division 2 times.
    """
    def __init__(self, op='attsum', activation='tanh', init_stdev=0.01, **kwargs):
        """
        Basic Constructor
        """
        self.supports_masking = True
        assert op in {'attsum', 'attmean'}
        assert activation in {None, 'tanh'}
        self.op = op
        self.activation = activation
        self.init_stdev = init_stdev
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Define the variables and the trainable weights
        """
        init_val_v = (np.random.randn(input_shape[2]) * self.init_stdev).astype(K.floatx())
        self.att_v = K.variable(init_val_v, name='att_v')
        init_val_W = (np.random.randn(input_shape[2], input_shape[2]) * self.init_stdev).astype(K.floatx())
        self.att_W = K.variable(init_val_W, name='att_W')
        self.trainable_weights = [self.att_v, self.att_W]
    
    def call(self, x, mask=None):
        """
        The main function in this class
        Weight matrix 'W' adds complexity and tanh adds non-linearity
        Basic attention can be done with just weight vector 'v'

        The individual weight for each word is saved in the attribute self.att_weights
        """
        y = K.dot(x, self.att_W)
        if not self.activation:
            weights = K.theano.tensor.tensordot(self.att_v, y, axes=[0, 2])
        elif self.activation == 'tanh':
            weights = K.theano.tensor.tensordot(self.att_v, K.tanh(y), axes=[0, 2])
        weights = K.softmax(weights)
        
        self.att_weights = weights # save the weights of each word

        out = x * K.permute_dimensions(K.repeat(weights, x.shape[2]), [0, 2, 1])
        if self.op == 'attsum':
            out = out.sum(axis=1)
        elif self.op == 'attmean':
            out = out.sum(axis=1) / mask.sum(axis=1, keepdims=True)
        return K.cast(out, K.floatx())

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])
    
    def compute_mask(self, x, mask):
        return None
    
    def get_config(self):
        config = {'op': self.op, 'activation': self.activation, 'init_stdev': self.init_stdev}
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MeanOverTime(Layer):
    """
    Mean over time layer (mean pooling through the whole sentence length)
    Supports masking (does not average the masked vectors)
    """
    def __init__(self, mask_zero=True, **kwargs):
        self.mask_zero = mask_zero
        self.supports_masking = True
        super(MeanOverTime, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if self.mask_zero:
            return K.cast(x.sum(axis=1) / mask.sum(axis=1, keepdims=True), K.floatx())
        else:
            return K.mean(x, axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])
    
    def compute_mask(self, x, mask):
        return None
    
    def get_config(self):
        config = {'mask_zero': self.mask_zero}
        base_config = super(MeanOverTime, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Conv1DWithMasking(Convolution1D):
    """
    Convolution 1D with Masking
    """
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(Conv1DWithMasking, self).__init__(**kwargs)
    
    def compute_mask(self, x, mask):
        return mask


class MaxPooling1DWithMasking(Layer):
    """
    MaxPooling1D that supports masking
    Not properly implemented yet
    Intended to use with very deep CNN (because multi-layer CNN usually uses MaxPooling)
    """
    def __init__(self, pool_length=2, stride=None,
                border_mode='valid', **kwargs):
        super(MaxPooling1DWithMasking, self).__init__(**kwargs)
        self.supports_masking = True
        if stride is None:
            stride = pool_length
        self.pool_length = pool_length
        self.stride = stride
        self.st = (self.stride, 1)
        self.pool_size = (pool_length, 1)
        assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
        self.border_mode = border_mode
        self.input_spec = [InputSpec(ndim=3)]

    def get_output_shape_for(self, input_shape):
        length = conv_output_length(input_shape[1], self.pool_length,
                                    self.border_mode, self.stride)
        return (input_shape[0], length, input_shape[2])

    def _pooling_function(self, inputs, pool_size, strides,
                        border_mode, dim_ordering):
        output = K.pool2d(inputs, pool_size, strides,
                        border_mode, dim_ordering, pool_mode='max')
        return output

    def call(self, x, mask=None):
        x = K.expand_dims(x, -1)   # add dummy last dimension
        x = K.permute_dimensions(x, (0, 2, 1, 3))
        output = self._pooling_function(inputs=x, pool_size=self.pool_size,
                                        strides=self.st,
                                        border_mode=self.border_mode,
                                        dim_ordering='th')
        output = K.permute_dimensions(output, (0, 2, 1, 3))
        return K.squeeze(output, 3)  # remove dummy last dimension

    def get_config(self):
        config = {'stride': self.stride,
                'pool_length': self.pool_length,
                'border_mode': self.border_mode}
        base_config = super(MaxPooling1DWithMasking, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
