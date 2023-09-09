from backbone import *
from custom.CustomLayers import Padding2D, ConvBlock, ResBlock


class ShuffleNet(models.Model):
    def __init__(self, **kwargs):
        super(ShuffleNet, self).__init__(**kwargs)
        self.padding = Padding2D(pad_size=(1, 1))
        self.conv_block = ConvBlock(filters=16,
                                    kernel_size=(3, 3),
                                    strides=(2, 2),
                                    padding="VALID")

        self.grouped_res_block1 = ResBlock(expansion=1.,
                                           source_size=16, embed_size=16,
                                           kernel_size=(3, 3), strides=(1, 1),
                                           padding="VALID", groups=1)

        self.grouped_res_block2 = models.Sequential([ResBlock(expansion=2.,
                                                              source_size=16, embed_size=24,
                                                              kernel_size=(3, 3), strides=(2, 2),
                                                              padding="VALID", groups=1),
                                                     ResBlock(expansion=2.,
                                                              source_size=40, embed_size=40,
                                                              kernel_size=(3, 3), strides=(1, 1),
                                                              padding="VALID", groups=1)])

        self.grouped_res_block3 = models.Sequential([ResBlock(expansion=2.,
                                                              source_size=40, embed_size=56,
                                                              kernel_size=(3, 3), strides=(2, 2),
                                                              padding="VALID", groups=2),
                                                     *[ResBlock(expansion=2.,
                                                                source_size=96, embed_size=96,
                                                                kernel_size=(3, 3), strides=(1, 1),
                                                                padding="VALID", groups=2)
                                                       for i in range(2)]])

        self.grouped_res_block4 = models.Sequential([ResBlock(expansion=3.,
                                                              source_size=96, embed_size=128,
                                                              kernel_size=(3, 3), strides=(2, 2),
                                                              padding="VALID", groups=4),
                                                     *[ResBlock(expansion=3.,
                                                                source_size=224, embed_size=224,
                                                                kernel_size=(3, 3), strides=(1, 1),
                                                                padding="VALID", groups=4)
                                                       for i in range(5)]])

        self.grouped_res_block5 = models.Sequential([ResBlock(expansion=3.,
                                                              source_size=224, embed_size=288,
                                                              kernel_size=(3, 3), strides=(2, 2),
                                                              padding="VALID", groups=8),
                                                     *[ResBlock(expansion=3.,
                                                                source_size=512, embed_size=512,
                                                                kernel_size=(3, 3), strides=(1, 1),
                                                                padding="VALID", groups=8)
                                                       for i in range(2)]])
    def call(self, inputs, *args, **kwargs):

        x = self.padding(inputs)
        x = self.conv_block(x)

        x = self.grouped_res_block1(x)
        x = self.grouped_res_block2(x)
        P3 = self.grouped_res_block3(x)

        P4 = self.grouped_res_block4(P3)
        P5 = self.grouped_res_block5(P4)

        return P5, P4, P3
