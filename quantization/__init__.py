import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models

def layer_judge(layer, fp_layer):
    fp_layer.kernel = tf.Variable(initial_value=tf.cast(layer.kernel,
                                                        tf.float16))
    try:
        if fp_layer.bias is not None:
            fp_layer.bias = tf.Variable(initial_value=tf.cast(layer.bias,
                                                              tf.float16))
    except Exception as e:
        print(repr(e))


def norm_judge(layer, fp_layer):
    fp_layer.gamma = tf.Variable(initial_value=tf.cast(layer.gamma,
                                                       tf.float32))
    fp_layer.beta = tf.Variable(initial_value=tf.cast(layer.beta,
                                                      tf.float32))


