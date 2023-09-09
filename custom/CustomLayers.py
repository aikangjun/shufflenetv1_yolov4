from custom import *
from custom.utils import swish
import tensorflow as tf

class Padding2D(layers.Layer):
    def __init__(self,
                 pad_size: tuple = None,
                 **kwargs):
        super(Padding2D, self).__init__(**kwargs)
        assert pad_size.__len__() == 2
        self.pad_size = pad_size
        self.padding = [[0, 0],
                        [pad_size[0], pad_size[0]],
                        [pad_size[1], pad_size[1]],
                        [0, 0]]

    def get_config(self):
        config = super(Padding2D, self).get_config()
        config.update({
            'pad_size': self.pad_size
        })
        return config

    def call(self, input, padding_mode='CONSTANT', *args, **kwargs):
        assert padding_mode in ['CONSTANT', 'REFLECT', 'SYMMETRIC']

        return tf.pad(input, self.padding, mode=padding_mode)


class ShuffleUnit(layers.Layer):
    """
    Feature shuffling
    """

    def __init__(self,
                 groups: int,
                 **kwargs):
        super(ShuffleUnit, self).__init__(**kwargs)
        self.groups = groups

    def get_config(self):
        config = super(ShuffleUnit, self).get_config()
        config.update({
            'groups': self.groups
        })
        return config

    def call(self, input, *args, **kwargs):
        input_size = tf.shape(input)
        x = tf.reshape(input, shape=[input_size[0], input_size[1], input_size[2], self.groups, -1])
        x = tf.transpose(x, perm=[0, 1, 2, 4, 3])
        x = tf.reshape(x, shape=[input_size[0], input_size[1], input_size[2], input.shape[-1]])

        return x


class GroupConv2D(layers.Conv2D):
    def __init__(self,
                 groups: int = None,
                 **kwargs):
        super(GroupConv2D, self).__init__(**kwargs)
        self.built = None
        self.groups = groups
        self.padding = self.padding.upper()

    def get_config(self):

        config = super(GroupConv2D, self).get_config()
        config.update({
            'groups': self.groups
        })

        return config

    def build(self, input_shape):

        if not input_shape[-1]:
            raise ValueError('The channel dimension of the inputs '
                             'should be [N, H, W, C]. Found `None`.')

        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size,) * 2

        kernel_shape = self.kernel_size + (input_shape[-1] // self.groups, self.filters)

        self.kernel = self.add_weight(name='kernel',
                                      trainable=True,
                                      shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      dtype=self.dtype)

        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        trainable=True,
                                        shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        dtype=self.dtype)
        else: self.bias = None

        self.built = True

    def call(self, input, *args, **kwargs):

        inputs = tf.split(input, num_or_size_splits=self.groups, axis=-1)
        kernels = tf.split(self.kernel, num_or_size_splits=self.groups, axis=-1)

        feats = [tf.nn.conv2d(input, filters=kernels[i], strides=self.strides,
                              padding=self.padding)
                 for i, input in enumerate(inputs)]

        output = tf.concat(feats, axis=-1)

        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)

        if self.activation:
            output = self.activation(output)

        return output


class ConvBlock(models.Sequential):
    def __init__(self,
                 filters: int,
                 kernel_size: int or tuple,
                 strides: tuple = (1, 1),
                 padding: str = "SAME",
                 use_bias: bool = False,
                 activate: bool = True,
                 upsample: bool = False,
                 normalize: bool = True,
                 kernel_initializer: str = "random_normal",
                 kernel_regularizer: str = "l2",
                 **kwargs):
        assert padding in ["SAME", "VALID"]
        kernel_initializer = initializers.get(kernel_initializer)
        kernel_regularizer = regularizers.get(kernel_regularizer)

        sequentials = []
        sequentials.append(layers.Conv2D(filters=filters,
                                         kernel_size=kernel_size,
                                         strides=strides,
                                         padding=padding,
                                         use_bias=use_bias,
                                         kernel_initializer=kernel_initializer,
                                         kernel_regularizer=kernel_regularizer))
        if normalize: sequentials.append(layers.BatchNormalization())
        if activate: sequentials.append(layers.Activation(tf.nn.leaky_relu))
        if upsample: sequentials.append(layers.UpSampling2D(interpolation='bilinear'))
        super(ConvBlock, self).__init__(layers=sequentials, **kwargs)


class ResBlock(layers.Layer):
    def __init__(self,
                 expansion: float,
                 source_size: int,
                 embed_size: int,
                 kernel_size: tuple,
                 strides: tuple,
                 padding: str = "VALID",
                 groups: int = 1,
                 kernel_initializer: str = "random_normal",
                 kernel_regularizer: str = "l2",
                 **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        assert padding in ["SAME", "VALID"]
        kernel_initializer = initializers.get(kernel_initializer)
        kernel_regularizer = regularizers.get(kernel_regularizer)

        self.init_grouped_conv = GroupConv2D(filters=int(expansion * source_size),
                                             kernel_initializer=kernel_initializer,
                                             kernel_regularizer=kernel_regularizer,
                                             kernel_size=(1, 1),
                                             strides=(1, 1),
                                             use_bias=False,
                                             groups=groups)

        self.depthwise_conv = layers.DepthwiseConv2D(depthwise_initializer=kernel_initializer,
                                                     depthwise_regularizer=kernel_regularizer,
                                                     kernel_size=kernel_size,
                                                     strides=strides,
                                                     padding=padding,
                                                     use_bias=False)

        self.final_grouped_conv = GroupConv2D(filters=embed_size,
                                              kernel_initializer=kernel_initializer,
                                              kernel_regularizer=kernel_regularizer,
                                              kernel_size=(1, 1),
                                              strides=(1, 1),
                                              use_bias=False,
                                              groups=groups)

        self.init_batch_norm = layers.BatchNormalization()
        self.middle_batch_norm = layers.BatchNormalization()
        self.final_batch_norm = layers.BatchNormalization()

        if padding == "VALID":
            self.padding = Padding2D(pad_size=(kernel_size[0] // 2,
                                               kernel_size[1] // 2))
        if strides != (1, 1):
            self.pooling = layers.MaxPooling2D(pool_size=kernel_size,
                                               strides=strides,
                                               padding=padding)

    def call(self, inputs, *args, **kwargs):

        x = self.init_grouped_conv(inputs)
        x = self.init_batch_norm(x)
        x = tf.nn.leaky_relu(x)

        if hasattr(self, "padding"):
            x = self.padding(x)
        x = self.depthwise_conv(x)
        x = self.middle_batch_norm(x)
        x = tf.nn.leaky_relu(x)

        x = self.final_grouped_conv(x)
        x = self.final_batch_norm(x)

        if hasattr(self, "pooling") and hasattr(self, "padding"):
            inputs = self.padding(inputs)
            inputs = self.pooling(inputs)
            x = tf.concat([inputs, x], axis=-1)
        else:
            x = x + inputs

        x = tf.nn.leaky_relu(x)

        return x


class SPPBlock(layers.Layer):
    def __init__(self, **kwargs):
        super(SPPBlock, self).__init__(**kwargs)

        self.pool5x5 = layers.MaxPooling2D(pool_size=(5, 5),
                                           strides=(1, 1),
                                           padding='SAME')
        self.pool9x9 = layers.MaxPooling2D(pool_size=(9, 9),
                                           strides=(1, 1),
                                           padding='SAME')
        self.pool13x13 = layers.MaxPooling2D(pool_size=(13, 13),
                                             strides=(1, 1),
                                             padding='SAME')

    def call(self, inputs, *args, **kwargs):
        p5x5 = self.pool5x5(inputs)
        p9x9 = self.pool9x9(inputs)
        p13x13 = self.pool13x13(inputs)

        x = tf.concat([p13x13, p9x9, p5x5, inputs], axis=-1)

        return x


class MultiConvBlock(layers.Layer):
    def __init__(self,
                 filters: list,
                 num_blocks: int,
                 kernel_sizes: list,
                 strides: tuple = (1, 1),
                 padding: str = "SAME",
                 use_bias: bool = False,
                 activate: bool = True,
                 normalize: bool = True,
                 **kwargs):
        super(MultiConvBlock, self).__init__(**kwargs)

        self.modules = [ConvBlock(filters=filters[i],
                                  kernel_size=kernel_sizes[i],
                                  strides=strides,
                                  padding=padding)
                        for i in range(num_blocks)]

    def call(self, x, *args, **kwargs):
        for module in self.modules:
            x = module(x)

        return x

tf.nn.sigmoid
class BottleNeck(models.Model):
    def __init__(self, **kwargs):
        super(BottleNeck, self).__init__(**kwargs)

        self.spp = SPPBlock()

        self.P5_in_conv = MultiConvBlock(filters=[192, 384, 192],
                                         num_blocks=3,
                                         kernel_sizes=[1, 3, 1])
        self.P5_up_conv = MultiConvBlock(filters=[384, 512, 384],
                                         num_blocks=3,
                                         kernel_sizes=[1, 3, 1])
        self.P5_upsample = ConvBlock(filters=224,
                                     kernel_size=(1, 1),
                                     upsample=True)

        self.P4_in_conv = ConvBlock(filters=224,
                                    kernel_size=(1, 1))
        self.P4_up_conv = MultiConvBlock(filters=[224, 448, 224, 448, 224],
                                         num_blocks=5,
                                         kernel_sizes=[1, 3, 1, 3, 1])
        self.P4_upsample = ConvBlock(filters=128,
                                     kernel_size=(1, 1),
                                     upsample=True)

        self.P3_in_conv = ConvBlock(filters=128,
                                    kernel_size=(1, 1))
        self.P3_conv = MultiConvBlock(filters=[128, 256, 128, 256, 128],
                                      num_blocks=5,
                                      kernel_sizes=[1, 3, 1, 3, 1])

        self.P3_downsample = ConvBlock(filters=224,
                                       kernel_size=(1, 1),
                                       strides=(2, 2))
        self.P4_out_conv = MultiConvBlock(filters=[224, 448, 224, 448, 224],
                                          num_blocks=5,
                                          kernel_sizes=[1, 3, 1, 3, 1])

        self.P4_downsample = ConvBlock(filters=384,
                                       kernel_size=(1, 1),
                                       strides=(2, 2))
        self.P5_out_conv = MultiConvBlock(filters=[384, 512, 368, 512, 384],
                                          num_blocks=5,
                                          kernel_sizes=[1, 3, 1, 3, 1])

    def call(self, inputs, *args, **kwargs):
        P5, P4, P3 = inputs

        P5 = self.P5_in_conv(P5)
        P5 = self.spp(P5)
        P5 = self.P5_up_conv(P5)
        P5_upsample = self.P5_upsample(P5)

        P4 = self.P4_in_conv(P4)
        P4 = tf.concat([P4, P5_upsample], axis=-1)
        P4 = self.P4_up_conv(P4)
        P4_upsample = self.P4_upsample(P4)

        P3 = self.P3_in_conv(P3)
        P3 = tf.concat([P3, P4_upsample], axis=-1)
        P3 = self.P3_conv(P3)

        P3_downsample = self.P3_downsample(P3)
        P4 = tf.concat([P3_downsample, P4], axis=-1)
        P4 = self.P4_out_conv(P4)

        P4_downsample = self.P4_downsample(P4)
        P5 = tf.concat([P4_downsample, P5], axis=-1)
        P5 = self.P5_out_conv(P5)

        return P5, P4, P3


class DetectHead(models.Model):
    def __init__(self,
                 layers_num: int,
                 anchors_num: int,
                 classes_num: int,
                 kernel_initializer: str = "random_normal",
                 kernel_regularizer: str = "l2",
                 **kwargs):
        super(DetectHead, self).__init__(**kwargs)
        feats_name = ["P5", "P4", "P3"]

        kernel_initializer = initializers.get(kernel_initializer)
        kernel_regularizer = regularizers.get(kernel_regularizer)

        self.conv_layers = [layers.Conv2D(filters=anchors_num * (4 + 1 + classes_num),
                                          kernel_size=(1, 1),
                                          kernel_initializer=kernel_initializer,
                                          kernel_regularizer=kernel_regularizer,
                                          name=feats_name[i],
                                          use_bias=False)
                            for i in range(layers_num)]

    def call(self, inputs, *args, **kwargs):
        outputs = []
        for input, layer in zip(inputs, self.conv_layers):
            output = layer(input)
            outputs.append(output)

        return outputs
