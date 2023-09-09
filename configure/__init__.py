# ===generator===
train_split = 0.9
input_size = (416, 416)
max_boxes = 100

# ===training===
Epoches = 30
batch_size = 8
learning_rate = 1e-4
warmup_learning_rate = 1e-5
min_learning_rate = 1e-6
cosine_scheduler = True

# ===prediction===
iou = 0.2
score = 0.2
generate_step = 100

# ===drawing===
font_color = (0, 255, 0)
rect_color = (0, 0, 255)
thickness = 0.5
