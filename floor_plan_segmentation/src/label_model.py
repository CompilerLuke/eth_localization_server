import torch
import cv2
import numpy as np
from dataclasses import dataclass
import yaml

@dataclass
class Label_Model_Args(yaml.YAMLObject):
    min_confidence_ratio: float = 2
    pad: float = 0.7
    digit_thr: float = 0.5


class Label_Model:
    def __init__(self, device, digit_model, args):
        self.digit_model = digit_model
        self.device = device
        self.args = args

    def predict_digit(self, dig):
        if (dig == 0).all():
            return -1

        if (dig[0:14, :] == 0).all():  # todo: fix hack recognition of .
            return 10

        t = torch.tensor(dig.reshape(1, 1, 28, 28)).to(self.device)
        pred = self.digit_model(t)[0].cpu()

        values, indices = torch.topk(pred, 2)

        ratio = torch.exp(values[0] - values[1]).item()

        if ratio < self.args.min_confidence_ratio:
            return -1
        else:
            return indices[0].item()


    def split(self, patch):
        patch = 255 - patch
        thr = cv2.threshold(patch, 80, 255, cv2.THRESH_OTSU)[1]

        n, labels = cv2.connectedComponents(thr)
        islands = [labels == i for i in range(1, n)]

        images = []
        xs = []

        (x1, y1, w1, h1) = cv2.boundingRect(thr)

        for island in islands:
            (x0, y0, w, h) = cv2.boundingRect(island.astype(np.uint8))
            images.append(patch[y1:y1 + h1, x0:x0 + w])
            xs.append(x0)

        return [images[i] for i in np.argsort(xs)]


    def canonicalize_digit(self, digit):
        (h,w) = digit.shape
        digit = np.pad(digit, int(self.args.pad*w))
        digit = cv2.resize(digit, (28,28), interpolation=cv2.INTER_LINEAR)
        digit = digit.astype(np.float32) / 255.

        thr = self.args.digit_thr
        mask = digit > thr
        digit[~mask] = 0
        digit[mask] = 1
        digit = cv2.GaussianBlur(digit,(3,3),0.5)

        return digit

    def __call__(self, img) -> str:
        label = ""

        for digit_img in self.split(img):
            digit_img = self.canonicalize_digit(digit_img)
            digit = self.predict_digit(digit_img)

            if digit == -1:
                return ""

            label += self.digit_model.letters[digit]

        return label
