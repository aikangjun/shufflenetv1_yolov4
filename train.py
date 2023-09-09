import os
import numpy as np
import tensorflow as tf
from yolo import Yolo
from configure import *
from configure import config as cfg
from _utils.generate import Generator
from _utils.utils import WarmUpCosineDecayScheduler

if __name__ == '__main__':

    yolo_ = Yolo(anchors=cfg.anchors,
                 classes_name=cfg.category_names_to_detect,
                 learning_rate=learning_rate,
                 score_thresh=score,
                 iou_thresh=iou,
                 max_boxes=max_boxes)

    data_gen = Generator(annotation_path=r'.\data_info\annotation.txt',
                         input_size=input_size,
                         batch_size=batch_size,
                         train_ratio=train_split,
                         anchors=cfg.anchors,
                         num_class=cfg.num_classes)

    train_gen = data_gen.generate(training=True)
    valid_gen = data_gen.generate(training=False)

    if not os.path.exists(cfg.ckpt_path):
        os.makedirs(cfg.ckpt_path)

    ckpt = tf.train.Checkpoint(model=yolo_.model, optimizer=yolo_.optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, cfg.ckpt_path, max_to_keep=5)

    # 如果检查点存在，则恢复最新的检查点，加载模型
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print('Latest checkpoint restored!!')

    if cosine_scheduler:
        total_steps = data_gen.get_train_step() * Epoches
        warmup_steps = int(data_gen.get_train_step() * Epoches * 0.2)
        hold_steps = data_gen.get_train_step() * data_gen.batch_size
        reduce_lr = WarmUpCosineDecayScheduler(global_interval_steps=total_steps,
                                               warmup_interval_steps=warmup_steps,
                                               hold_interval_steps=hold_steps,
                                               learning_rate_base=learning_rate,
                                               warmup_learning_rate=warmup_learning_rate,
                                               min_learning_rate=min_learning_rate,
                                               verbose=0)
    for epoch in range(Epoches):
        # ----training----
        print('------start training------')
        for i in range(data_gen.get_train_step()):
            sources, targets = next(train_gen)
            if cosine_scheduler:
                learning_rate = reduce_lr.batch_begin()
                yolo_.optimizer.learning_rate = learning_rate
            yolo_.train(sources, targets)
            if not (i + 1) % generate_step:
                yolo_.generate_sample(sources, i + 1)
                print('yolo_loss: {}\n'.format(yolo_.train_loss.result().numpy()),
                      'conf_acc: {}\n'.format(yolo_.train_conf.result().numpy() * 100),
                      'class_acc: {}\n'.format(yolo_.train_class_acc.result().numpy() * 100))

        # ----validating----
        print('------start validating------')
        for i in range(data_gen.get_val_step()):
            sources, targets = next(valid_gen)
            yolo_.validate(sources, targets)
            if not (i + 1) % generate_step:
                print('yolo_loss: {}\n'.format(yolo_.valid_loss.result().numpy()),
                      'conf_acc: {}\n'.format(yolo_.valid_conf.result().numpy() * 100),
                      'class_acc: {}\n'.format(yolo_.valid_class_acc.result().numpy() * 100))

        print(f'Epoch {epoch + 1}\n',
              f'train_yolo_loss: {yolo_.train_loss.result().numpy()}\n',
              f'train_conf_acc: {yolo_.train_conf.result().numpy() * 100}\n',
              f'train_class_acc: {yolo_.train_class_acc.result().numpy() * 100}\n',
              f'valid_yolo_loss: {yolo_.valid_loss.result().numpy()}\n',
              f'valid_conf_acc: {yolo_.valid_conf.result().numpy() * 100}\n',
              f'valid_class_acc: {yolo_.valid_class_acc.result().numpy() * 100}\n')

        ckpt_save_path = ckpt_manager.save()

        yolo_.train_loss.reset_states()
        yolo_.valid_loss.reset_states()

        yolo_.train_conf.reset_states()
        yolo_.valid_conf.reset_states()

        yolo_.train_class_acc.reset_states()
        yolo_.valid_class_acc.reset_states()
