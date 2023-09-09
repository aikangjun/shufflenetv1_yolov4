import numpy as np
from _utils.utils import get_random_data, preprocess_true_boxes
import configure.config as cfg

class Generator:
    def __init__(self,
                 annotation_path: str,
                 input_size: tuple,
                 batch_size: int,
                 train_ratio: float,
                 anchors: np.ndarray,
                 num_class: int):
        self.annotation_path = annotation_path
        self.input_size = input_size
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.anchors = anchors
        self.num_class = num_class
        self.split_train_val()

    def split_train_val(self):
        with open(self.annotation_path) as f:
            # lines 为list,内部元素为str
            lines = f.readlines()
        np.random.shuffle(lines)
        num_train = int(lines.__len__() * self.train_ratio)
        self.train_lines = lines[:num_train]
        self.val_lines = lines[num_train:]

    def get_train_step(self):
        if not self.train_lines.__len__() % self.batch_size:
            return self.train_lines.__len__() // self.batch_size
        else:
            return self.train_lines.__len__() // self.batch_size + 1

    def get_val_step(self):
        if not self.val_lines.__len__() % self.batch_size:
            return self.val_lines.__len__() // self.batch_size
        else:
            return self.val_lines.__len__() // self.batch_size

    def generate(self, training: bool = True):
        while True:
            sources, targets = [], []
            if training:
                lines = self.train_lines
            if not training:
                lines = self.val_lines
            for i, line in enumerate(lines):
                image_data, box_data = get_random_data(line,
                                                       image_size=self.input_size,
                                                       max_boxes=cfg.max_boxes,
                                                       random=False)
                sources.append(image_data)
                targets.append(box_data)

                if sources.__len__() == self.batch_size or i == lines.__len__() - 1:
                    anno_sources = np.array(sources.copy())
                    anno_targets = preprocess_true_boxes(
                        np.array(targets.copy()), self.input_size, self.anchors,
                        self.num_class)

                    sources.clear()
                    targets.clear()
                    # 返回的anno_sources数据类型为numpy.ndarray 形状为(bacth_szie,416,416,3)
                    # 返回的anno_targets数据类型为list,长度为3。内部元素类型为numpy.ndarray,
                    # 其形状分别为(batch_size,13,13,3,5+num_class)(batch_size,26,26,3,5+num_class)
                    # (batch_size,52,52,3,5_num_class)
                    yield anno_sources,anno_targets


if __name__ == '__main__':
    import configure.config as cfg
    f = open(cfg.data_dir+'\\annotation.txt')
    lines = f.readlines()
    print(lines[-1])
    line = lines[-1].split('\t')
    # print(lines)

    gen = Generator(annotation_path=cfg.data_dir+'\\annotation.txt',
                    input_size=cfg.image_size,
                    batch_size=4,
                    train_ratio=0.7,
                    anchors=cfg.anchors,
                    num_class=cfg.num_classes)
    train_gen = gen.generate(training=True)
    source,target = next(train_gen)
    print(source.shape)
    print(type(source))
    print(type(target[0]))
    print(target[0].shape)
