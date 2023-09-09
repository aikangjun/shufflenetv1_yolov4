from network import *


def yolo_head(feats, anchors, num_classes, input_shape):
    num_anchors = len(anchors)
    # ---------------------------------------------------#
    #   [1, 1, 1, num_anchors, 2]
    # ---------------------------------------------------#
    anchors_tensor = tf.reshape(tf.cast(anchors, dtype=tf.float32), [1, 1, 1, num_anchors, 2])

    # ---------------------------------------------------#
    #   获得x，y的网格
    #   (13, 13, 1, 2)
    # ---------------------------------------------------#
    grid_shape = tf.shape(feats)[1:3]
    grids = tf.meshgrid(tf.range(grid_shape[1]), tf.range(grid_shape[0]))
    grid_xy = tf.stack(grids, axis=-1)
    grid = grid_xy[..., tf.newaxis, :]
    grid = tf.cast(grid, dtype=feats.dtype)

    # ---------------------------------------------------#
    #   将预测结果调整成(batch_size,13,13,3,85)
    #   85可拆分成4 + 1 + 80
    #   4代表的是中心宽高的调整参数
    #   1代表的是框的置信度
    #   80代表的是种类的置信度
    # ---------------------------------------------------#
    feats = tf.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # ---------------------------------------------------#
    #   将预测值调成真实值
    #   box_xy对应框的中心点
    #   box_wh对应框的宽和高
    # ---------------------------------------------------#
    box_xy = (tf.sigmoid(feats[..., :2]) + grid) / tf.cast(grid_shape[..., ::-1], dtype=feats.dtype)
    box_wh = tf.exp(feats[..., 2:4]) * anchors_tensor / tf.cast(input_shape[..., ::-1], dtype=feats.dtype)
    box_confidences = tf.sigmoid(feats[..., 4:5])
    box_class_probs = tf.nn.softmax(feats[..., 5:], axis=-1)

    # ---------------------------------------------------------------------#
    #   在计算loss的时候返回grid, feats, box_xy, box_wh
    #   在预测的时候返回box_xy, box_wh, box_confidence, box_class_probs
    # ---------------------------------------------------------------------#

    return box_xy, box_wh, box_confidences, box_class_probs


# ---------------------------------------------------#
#   对box进行调整，使其符合真实图片的样子
# ---------------------------------------------------#
def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    # -----------------------------------------------------------------#
    #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
    # -----------------------------------------------------------------#
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    input_shape = tf.cast(input_shape, dtype=box_yx.dtype)
    image_shape = tf.cast(image_shape, dtype=box_yx.dtype)

    new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))
    # -----------------------------------------------------------------#
    #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
    #   new_shape指的是宽高缩放情况
    # -----------------------------------------------------------------#
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = tf.concat([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)

    boxes *= tf.concat([image_shape, image_shape], axis=-1)
    return boxes


# ---------------------------------------------------#
#   获取每个box和它的得分
# ---------------------------------------------------#
def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape, letterbox_image):
    # -----------------------------------------------------------------#
    #   将预测值调成真实值
    #   box_xy : -1,13,13,3,2;
    #   box_wh : -1,13,13,3,2;
    #   box_confidence : -1,13,13,3,1;
    #   box_class_probs : -1,13,13,3,80;
    # -----------------------------------------------------------------#
    box_xy, box_wh, box_confidences, box_class_probs = yolo_head(feats, anchors, num_classes, input_shape)
    # -----------------------------------------------------------------#
    #   在图像传入网络预测前会进行letterbox_image给图像周围添加灰条
    #   因此生成的box_xy, box_wh是相对于有灰条的图像的
    #   我们需要对齐进行修改，去除灰条的部分。
    #   将box_xy、和box_wh调节成y_min,y_max,xmin,xmax
    # -----------------------------------------------------------------#
    if letterbox_image:
        boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    else:
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)

        input_shape = tf.cast(input_shape, dtype=box_yx.dtype)
        image_shape = tf.cast(image_shape, dtype=box_yx.dtype)

        boxes = tf.concat([
            box_mins[..., 0:1] * image_shape[0],  # y_min
            box_mins[..., 1:2] * image_shape[1],  # x_min
            box_maxes[..., 0:1] * image_shape[0],  # y_max
            box_maxes[..., 1:2] * image_shape[1]  # x_max
        ], axis=-1)
    # -----------------------------------------------------------------#
    #   获得最终得分和框的位置
    # -----------------------------------------------------------------#
    boxes = tf.reshape(boxes, [-1, 4])
    box_confidences = tf.squeeze(box_confidences, axis=-1)
    box_confidences = tf.reshape(box_confidences, [-1])
    box_class_probs = tf.reshape(box_class_probs, [-1, num_classes])

    return boxes, box_confidences, box_class_probs


# ---------------------------------------------------#
#   图片预测
# ---------------------------------------------------#
def yolo_eval(yolo_outputs,
              anchors,
              classes_num,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5,
              letterbox_image=False):
    # ---------------------------------------------------#
    #   获得特征层的数量，有效特征层的数量为3
    # ---------------------------------------------------#
    num_layers = len(yolo_outputs)
    # -----------------------------------------------------------#
    #   13x13的特征层对应的anchor是[142, 110], [192, 243], [459, 401]
    #   26x26的特征层对应的anchor是[36, 75], [76, 55], [72, 146]
    #   52x52的特征层对应的anchor是[12, 16], [19, 36], [40, 28]
    # -----------------------------------------------------------#
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    # -----------------------------------------------------------#
    #   这里获得的是输入图片的大小，一般是416x416
    # -----------------------------------------------------------#
    batch_size = tf.shape(yolo_outputs[0])[0]
    input_shape = tf.shape(yolo_outputs[0])[1:3] * 32
    total_boxes = list()
    total_scores = list()
    total_classes = list()
    for i in range(batch_size):
        # -----------------------------------------------------------#
        #   遍历批量
        # -----------------------------------------------------------#
        boxes = list()
        box_scores = list()
        box_class_probs = list()
        for l in range(num_layers):
            # -----------------------------------------------------------#
            #   对每个特征层进行处理
            # -----------------------------------------------------------#
            _boxes, _box_confs, _box_class_probs = yolo_boxes_and_scores(yolo_outputs[l][i][tf.newaxis, ...],
                                                                         anchors[anchor_mask[l]], classes_num,
                                                                         input_shape, image_shape,
                                                                         letterbox_image)
            boxes.append(_boxes)
            box_scores.append(_box_confs)
            box_class_probs.append(_box_class_probs)
        # -----------------------------------------------------------#
        #   将每个特征层的结果进行堆叠
        # -----------------------------------------------------------#
        boxes = tf.concat(boxes, axis=0)
        box_scores = tf.concat(box_scores, axis=0)
        box_class_probs = tf.concat(box_class_probs, axis=0)

        # -----------------------------------------------------------#
        #   判断得分是否大于score_threshold
        # -----------------------------------------------------------#
        mask = box_scores >= score_threshold
        masked_scores = tf.boolean_mask(box_scores, mask)
        masked_boxes = tf.boolean_mask(boxes, mask)
        masked_probs = tf.boolean_mask(box_class_probs, mask)
        max_boxes_tensor = tf.cast(max_boxes, dtype=tf.int32)
        boxes_ = list()
        scores_ = list()
        classes_ = list()
        for c in range(classes_num):
            # -----------------------------------------------------------#
            #   取出所有box_scores >= score_threshold的框
            # -----------------------------------------------------------#
            class_boxes = tf.boolean_mask(masked_boxes, tf.equal(tf.argmax(masked_probs, axis=-1), c))
            class_box_scores = tf.boolean_mask(masked_scores, tf.equal(tf.argmax(masked_probs, axis=-1), c))

            # -----------------------------------------------------------#
            #   非极大抑制
            # -----------------------------------------------------------#
            nms_index = tf.image.non_max_suppression(
                class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)

            # -----------------------------------------------------------#
            #   框的位置，得分与种类
            # -----------------------------------------------------------#
            class_boxes = tf.gather(class_boxes, nms_index)
            class_box_scores = tf.gather(class_box_scores, nms_index)
            classes = tf.ones_like(class_box_scores, 'int32') * c
            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)
        boxes_ = tf.concat(boxes_, axis=0)
        scores_ = tf.concat(scores_, axis=0)
        classes_ = tf.concat(classes_, axis=0)
        total_boxes.append(boxes_)
        total_scores.append(scores_)
        total_classes.append(classes_)

    return total_boxes, total_scores, total_classes
