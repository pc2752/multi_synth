import config
import tensorflow as tf
import numpy as np

def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d"):
"""
output = deconv2d(input, output_shape)
Function for deconv, can be used for both 1D and 2D deconvolutional operations. 
Outputs a tensor of the shape defined in output_shape

Inputs:
input_ : A 4-d input tensor, of type [batch size, time dimension, frequency dimension, number of channels]
output_shape: a numpy array with 4 elements
k_h and k_w: filter dimensions in time and frequency
d_h and d_w: strides in time and frequency
stddev: Standard deviation to initialize filter kernels
name: scope name for the deconv operation
Outputs:
A 4-d tensor with shape fiven by output_shape

"""

  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))
    
    try:
      deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    # Support for verisons of TensorFlow before 0.7.0
    except AttributeError:
      deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())


    return deconv

def conv2d(inputs, num_filters, k_h=3, k_w=3,padding = 'same', strides = (1,1), stddev=0.02, name="conv2d" ):
  """
  Function for convolution, can be used to 2d and 1d convolutions. 
  """
    w = tf.get_variable('w', [k_h, k_w, num_filters, input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))  
    conv = tf.nn.conv2d(inputs,w, strides = strides, padding = padding, name = name)

     biases = tf.get_variable('biases', [num_filters], initializer=tf.constant_initializer(0.0))

     conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

     return conv

def downsample(inputs):
  """
  op for downsampling, will use conv2d probably.
  """
def upsample(inputs):
  """
  op for upsampling, may use deconv2d or image resize with bilinear or nearest neighbour interpolation.
  """

def selu(x):
  """
  The SELU non-linearity, used for the discriminator instead of RELU. See paper: https://arxiv.org/pdf/1711.06491.pdf
  Self normalizing neural network: https://arxiv.org/pdf/1706.02515.pdf.
  """
   alpha = 1.6732632423543772848170429916717
   scale = 1.0507009873554804934193349852946
   return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))


def bn(x):
  '''
   Batch normalization
   https://arxiv.org/abs/1502.03167
   '''
   return tf.layers.batch_normalization(x)

def gated_conv(inputs, conds, name):
    out_sig = tf.nn.sigmoid(conv2d(inputs, config.wavenet_filters, 3, 3, strides = (2,1), name = name+'_sig') + conds)
    out_tan = tf.nn.tanh(conv2d(inputs, config.wavenet_filters, 3, 3, strides = (2,1), name = name+'_tanh') + conds)
    out = tf.multiply(out_sig,out_tanh)    
    return out


def gated_deconv(inputs, conds, output_shape, name):
    out_sig = tf.nn.sigmoid(deconv2d(inputs, output_shape, 3, 3, strides = (2,1), name = name+'_sig') + conds)
    out_tan = tf.nn.tanh(deconv2d(inputs, output_shape, 3, 3, strides = (2,1), name = name+'_tanh') + conds)
    out = tf.multiply(out_tan, out_sig)
    return out




def encoder(inputs, conds, target_dimension, name = "Encoder"):
    input_dimension = inputs.get_shape()[1]
    num_convs = input_dimension/target_dimension

    outs = []

    out = gated_conv(inputs, conds, name = name + '_'+str(0))

    outs.append(out)

    for i in range(num_convs):
        out = gated_conv(out, conds,name = name +'_'+str(i+1) )
        outs.concat(out)
    return outs