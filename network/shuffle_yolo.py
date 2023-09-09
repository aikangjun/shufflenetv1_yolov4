from network import *
from backbone.shufflenet import ShuffleNet
from custom.CustomLayers import BottleNeck, DetectHead


class YOLO(models.Model):
    def __init__(self,
                 layers_num: int,
                 anchors_num: int,
                 classes_num: int,
                 **kwargs):
        super(YOLO, self).__init__(**kwargs)

        self.backbone = ShuffleNet()
        self.bottle_neck = BottleNeck()

        self.d_head = DetectHead(layers_num=layers_num,
                                 anchors_num=anchors_num,
                                 classes_num=classes_num)

    def call(self, inputs, *args, **kwargs):
        P5, P4, P3 = self.backbone(inputs)
        P5, P4, P3 = self.bottle_neck([P5, P4, P3])

        P5, P4, P3 = self.d_head([P5, P4, P3])

        return P5, P4, P3
