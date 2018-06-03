import tensorflow as tf

def atrous_conv1d(tensor, output_channels, is_causal=False, rate=1, pad='SAME', stddev=0.02, name="aconv1d"):
    """
        Args:
          tensor: A 3-D tensor.
          output_channels: An integer. Dimension of output channel.
          is_causal: A boolean. If true, apply causal convolution.
          rate: An integer. Dilation rate.
          pad: Either "SAME" or "VALID". If "SAME", make padding, else no padding.
          stddev: A float. Standard deviation for truncated normal initializer.
          name: A string. Name of scope.
        Returns:
          A tensor of the same shape as `tensor`, which has been
          processed through dilated convolution layer.
    """

    # Set filter size
    size = (2 if is_causal else 3)

    # Get input dimension
    in_dim = tensor.get_shape()[-1].value
    rate = [rate]

    with tf.variable_scope(name):
        # Make filter
        filter = tf.get_variable("w", [size, in_dim, output_channels],
                                 initializer=tf.truncated_normal_initializer(stddev=stddev))

        # Pre processing for dilated convolution
        if is_causal:
            # Causal convolution pre-padding
            if pad == 'SAME':
                pad_len = (size - 1) * rate
                x = tf.expand_dims(tf.pad(tensor, [[0, 0], [pad_len, 0], [0, 0]]),axis=1, name="X")
            else:
                x = tf.expand_dims(tensor, axis=1)
            # Apply 2d convolution
            out = tf.nn.atrous_conv2d(x, filter, rate=rate, padding='VALID')
        else:
            # Apply 2d convolution
            out = tf.nn.convolution(tensor,
                                      filter, dilation_rate=rate, padding=pad, data_format='NWC')
        # Reduce dimension
        # out = tf.squeeze(out, axis=1)

    return out

def conv1d(input_, output_channels, filter_width = 1, stride = 1, stddev=0.02, name = 'conv1d'):
    """
        Args:
          tensor: A 3-D tensor.
          output_channels: An integer. Dimension of output channel.
          filter_width: An integer. Size of filter.
          stride: An integer. Stride of convolution.
          stddev: A float. Standard deviation for truncated normal initializer.
          name: A string. Name of scope.
        Returns:
          A tensor of the shape as [batch size, timesteps, output channel], which has been
          processed through 1-D convolution layer.
    """

    # Get input dimension
    input_shape = input_.get_shape()
    input_channels = input_shape[-1].value

    with tf.variable_scope(name):
        # Make filter
        filter_ = tf.get_variable('w', [filter_width, input_channels, output_channels],
            initializer=tf.truncated_normal_initializer(stddev=stddev))

        # Convolution layer
        conv = tf.nn.conv1d(input_, filter_, stride = stride, padding = 'SAME')
        biases = tf.get_variable('biases', [output_channels], initializer=tf.constant_initializer(0.0))

        # Add bias
        conv = tf.nn.bias_add(conv, biases)

        return conv

def residual_block(input_, rate, scope="res"):

    input_dim = input_.get_shape()[-1].value

    with tf.variable_scope(scope):
        aconv_f = atrous_conv1d(input_,
                              output_channels=input_dim // 2,
                              rate=rate,
                              name="filter_aconv")
        aconv_g = atrous_conv1d(input_,
                              output_channels=input_dim // 2,
                              rate=rate,
                              name="gate_aconv")
        aconv = tf.multiply(aconv_f, tf.sigmoid(aconv_g))

        skip_connection = conv1d(aconv,
                                 output_channels=input_dim,
                                 name="skip_connection")
        res_output = conv1d(aconv,
                            output_channels=input_dim,
                            name="res_output")

        return skip_connection, res_output + input_

def skip_connection(tensor, logit_dim=256, use_mulaw=True):

    dim = tensor.get_shape()[-1].value

    with tf.variable_scope("last_skip_connection"):
        tensor = tf.nn.relu(tensor)
        tensor = conv1d(tensor, dim, name="conv1")
        tensor = tf.nn.relu(tensor)
        tensor = conv1d(tensor, logit_dim if use_mulaw else 1, name="conv2")
        return tensor


def mu_law_encode(audio, quantization_channels=256):
    '''Quantizes waveform amplitudes.'''
    with tf.name_scope('encode'):
        mu = tf.to_float(quantization_channels - 1)
        # Perform mu-law companding transformation (ITU-T, 1988).
        # Minimum operation is here to deal with rare large amplitudes caused
        # by resampling.
        safe_audio_abs = tf.minimum(tf.abs(audio), 1.0)
        magnitude = tf.log1p(mu * safe_audio_abs) / tf.log1p(mu)
        signal = tf.sign(audio) * magnitude
        # Quantize signal to the specified number of levels.
        return tf.one_hot(tf.to_int32((signal + 1) / 2 * mu + 0.5), quantization_channels)


def mu_law_decode(output, quantization_channels=256):
    '''Recovers waveform from quantized values.'''
    with tf.name_scope('decode'):
        mu = quantization_channels - 1
        # Map values back to [-1, 1].
        signal = 2 * (tf.to_float(output) / mu) - 1
        # Perform inverse of mu-law transformation.
        magnitude = (1 / mu) * ((1 + mu) ** abs(signal) - 1)
        return tf.sign(signal) * magnitude



