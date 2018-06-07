from module import *
import hyperparams as hp

def network(input_, use_mulaw=hp.use_mulaw):
    input_ = conv1d(input_, output_channels=hp.hidden_dim, filter_width=3)

    skip_connections = list()
    for i in hp.dilation:
        skip, res  = residual_block(input_, rate=i, scope="res_%d" % i)
        input_ = res
        skip_connections.append(skip)

    skip_output = tf.add_n(skip_connections)
    output = skip_connection(skip_output, use_mulaw=use_mulaw)

    return output

