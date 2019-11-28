# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import os
import re
from collections import defaultdict

from torch.utils.data import Dataset
from ret_benchmark.utils.img_reader import read_image


class BaseDataSet(Dataset):
    """
    Basic Dataset read image path from img_source
    img_source: list of img_path and label
    """

    def __init__(self, img_source, transforms=None, mode="RGB"):
        self.mode = mode
        self.transforms = transforms
        self.root = os.path.dirname(img_source)
        assert os.path.exists(img_source), f"{img_source} NOT found."
        self.img_source = img_source

        self.label_list = list()
        self.path_list = list()
        self.bboxes = dict()
        self._load_data()
        self.label_index_dict = self._build_label_index_dict()

    def __len__(self):
        return len(self.label_list)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"| Dataset Info |datasize: {self.__len__()}|num_labels: {len(self.label_list)}| bbox: {len(self.bboxes)} |"

    def _load_data(self):
        with open(self.img_source, 'r') as f:
            for line in f:
                _path, _label, *_rest = re.split(r",| ", line.strip())
                self.path_list.append(_path)
                self.label_list.append(_label)
                if len(_rest) == 4:  # Bounding box
                    self.bboxes[_path] = [int(v) for v in _rest]

    def _build_label_index_dict(self):
        index_dict = defaultdict(list)
        for i, label in enumerate(self.label_list):
            index_dict[label].append(i)
        return index_dict

    def __getitem__(self, index):
        path = self.path_list[index]
        img_path = os.path.join(self.root, path)
        label = self.label_list[index]

        img = read_image(img_path, mode=self.mode)
        if self.bboxes and self.bboxes.get(path):
            x1, y1, x2, y2 = self.bboxes[path]
            img = img.crop((x1, y1, x2, y2))
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label
