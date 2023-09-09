# tf的默认模型是通过pb文件保存
# 构造相同的模型，加载ckpt,再保存为tf模型
import tensorflow as tf
from yolo import Yolo
import configure.config as cfg
import tensorflow.keras as keras
from configure import *

layers = keras.layers
models = keras.models
# 只要两个模型具有相同的架构，它们就可以共享同一个检查点
yolo_ = Yolo(anchors=cfg.anchors,
             classes_name=cfg.category_names_to_detect,
             learning_rate=learning_rate,
             score_thresh=score,
             iou_thresh=iou,
             max_boxes=max_boxes)
ckpt = tf.train.Checkpoint(model=yolo_.model)
ckpt_manager = tf.train.CheckpointManager(checkpoint=ckpt,
                                          directory=cfg.ckpt_path,
                                          max_to_keep=5)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    print('Latest checkpoint restored!!')
# 低级API
# yolo_.network.save(filepath='.\\tf_model', save_format='tf')
# 高级API,保存和序列化，函数式模型
# 无状态层不会改变权重，因此即便存在额外的/缺失的无状态层，模型也可以具有兼容架构。
input = layers.Input(shape=(None, None, 3))
output = yolo_.model(input)
model = models.Model(input, output)
model.save(filepath='.\\tf_models', save_format='tf')
