from _utils.utils import get_anchors

# parse json file

train_annotation_path = (r'D:\dataset\image\COCO'
                         r'\annotations\instances_train2017.json')
validation_annotation_path = (r'D:\dataset\image\COCO'
                              r'\annotations\instances_val2017.json')

max_boxes = 20
category_names_to_detect = ['person', 'truck', 'boat']
# category_names_to_detect = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
#                             'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
#                             'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
#                             'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#                             'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
#                             'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#                             'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
#                             'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'TV', 'laptop', 'mouse', 'remote',
#                             'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
#                             'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
num_classes = category_names_to_detect.__len__()
num_anchor = 3

data_dir = r'C:\Users\chen\Desktop\zvan\shufflenetv1_yolov4\data_info'
image_dir = r'D:\dataset\image\COCO\image\train2017'
image_size = (416, 416)  # (h,w)
# kmeans_for_anchors

# model
anchors = get_anchors(anchors_path=r'C:\Users\chen\Desktop\zvan\shufflenetv1_yolov4\data_info\anchors.txt')
batch_size = 8
ckpt_path = '.\\ckpt'

# lr
Epoches = 64
learning_rate = 1e-5
warmup_learning_rate = 1e-6
min_learning_rate = 1e-7
cosine_scheduler = False

iou = 0.6
score = 0.6
generate_step = 100

# predict
font_path = '.\\font\\simhei.ttf'

# draw
# ===drawing===
font_color = (0, 255, 0)
rect_color = (0, 0, 255)
thickness = 0.5
sample_path = 'result\\Batch{}.jpg'
# tf_model
tf_model_path = '.\\tf_model'
