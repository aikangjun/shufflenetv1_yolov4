# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from network.utils import yolo_eval
from network.shuffle_yolo import YOLO
from custom.loss_utils import yolo_loss
from configure import *
from configure import config as cfg


class Yolo:
    def __init__(self,
                 anchors: np.ndarray,
                 classes_name: list,
                 learning_rate: float,
                 score_thresh: float,
                 iou_thresh: float,
                 max_boxes: int):

        self.anchors = anchors
        self.max_boxes = max_boxes
        self.iou_thresh = iou_thresh
        self.score_thresh = score_thresh
        self.classes_name = classes_name
        self.anchors_num = anchors.__len__()
        self.classes_num = classes_name.__len__()
        self.learning_rate = learning_rate

        # model structure
        self.model = YOLO(layers_num=self.anchors_num // 3,
                          anchors_num=3,
                          classes_num=self.classes_num)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.train_loss = tf.keras.metrics.Mean()
        self.valid_loss = tf.keras.metrics.Mean()
        self.train_conf = tf.keras.metrics.BinaryAccuracy()
        self.valid_conf = tf.keras.metrics.BinaryAccuracy()
        self.train_class_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        self.valid_class_acc = tf.keras.metrics.SparseCategoricalAccuracy()

    # @tf.function
    def train(self, sources, targets):
        with tf.GradientTape() as tape:
            logits = self.model(sources)
            loss = yolo_loss(targets, logits, self.anchors, self.classes_num)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        logits = [tf.reshape(logit, shape=[tf.shape(logit)[0], tf.shape(logit)[1], tf.shape(logit)[2],
                                           self.anchors_num // 3, -1]) for logit in logits]

        prob_confs = [tf.sigmoid(logit[..., 4:5]) for logit in logits]
        real_confs = [target[..., 4:5] for target in targets]
        [self.train_conf(real_conf, prob_conf) for real_conf, prob_conf
         in zip(real_confs, prob_confs)]

        object_masks = [tf.squeeze(tf.cast(real_conf, dtype=tf.bool), axis=-1)
                        for real_conf in real_confs]
        prob_classes = [tf.boolean_mask(logit[..., 5:], mask)
                        for logit, mask in zip(logits, object_masks)]
        real_classes = [tf.boolean_mask(target[..., 5:], mask)
                        for target, mask in zip(targets, object_masks)]
        [self.train_class_acc(tf.argmax(real_class, axis=-1), prob_class)
         for real_class, prob_class in zip(real_classes, prob_classes)]

    @tf.function
    def validate(self, sources, targets):

        logits = self.model(sources)
        loss = yolo_loss(targets, logits, self.anchors, self.classes_num)

        self.valid_loss(loss)
        logits = [tf.reshape(logit, shape=[tf.shape(logit)[0], tf.shape(logit)[1], tf.shape(logit)[2],
                                           self.anchors_num // 3, -1]) for logit in logits]

        prob_confs = [tf.sigmoid(logit[..., 4:5]) for logit in logits]
        real_confs = [target[..., 4:5] for target in targets]
        [self.valid_conf(real_conf, prob_conf) for real_conf, prob_conf
         in zip(real_confs, prob_confs)]

        object_masks = [tf.squeeze(tf.cast(real_conf, dtype=tf.bool), axis=-1)
                        for real_conf in real_confs]
        prob_classes = [tf.boolean_mask(logit[..., 5:], mask)
                        for logit, mask in zip(logits, object_masks)]
        real_classes = [tf.boolean_mask(target[..., 5:], mask)
                        for target, mask in zip(targets, object_masks)]
        [self.valid_class_acc(tf.argmax(real_class, axis=-1), prob_class)
         for real_class, prob_class in zip(real_classes, prob_classes)]

    def generate_sample(self, sources, batch):
        """
        Drawing and labeling
        """
        logits = self.model(sources)
        image_size = tf.shape(sources)[1:3]
        out_boxes, out_scores, out_classes = yolo_eval(yolo_outputs=logits,
                                                       anchors=self.anchors,
                                                       classes_num=self.classes_num,
                                                       image_shape=image_size,
                                                       max_boxes=self.max_boxes,
                                                       score_threshold=self.score_thresh,
                                                       iou_threshold=self.iou_thresh)

        out_boxes = [out_box.numpy() for out_box in out_boxes]
        out_scores = [out_score.numpy() for out_score in out_scores]
        out_classes = [out_class.numpy() for out_class in out_classes]

        index = np.random.choice(np.shape(sources)[0], 1)[0]
        source = sources[index]
        image = Image.fromarray(np.uint8(source * 255))

        for i, coordinate in enumerate(out_boxes[index].astype('int')):
            left, top = list(reversed(coordinate[:2]))
            right, bottom = list(reversed(coordinate[2:]))

            font = ImageFont.truetype(font=cfg.font_path,
                                      size=np.floor(4e-2 * image.size[1] + 0.5).astype('int32'))

            label = '{:s}: {:.2f}'.format(self.classes_name[out_classes[index][i]],
                                          out_scores[index][i])

            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            draw.rectangle((left, top, right, bottom),
                           outline=rect_color, width=int(2 * thickness))

            draw.text(text_origin, str(label, 'UTF-8'),
                      fill=font_color, font=font)
            del draw
        image.save(cfg.sample_path.format(batch), quality=95, subsampling=0)
