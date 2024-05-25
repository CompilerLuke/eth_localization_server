from dataclasses import dataclass
from matplotlib import pyplot as plt
import cv2
import yaml
import numpy as np
import argparse
import json
import os

@dataclass
class SegmentationArgs(yaml.YAMLObject):
    contour_blur_kernel: int = 13
    contour_eps: float = 0.005
    outline_eps: float = 0.001
    outline_dilation: int = 21


class Segmentation_Model:
    def __init__(self, label_model, args=SegmentationArgs()):
        self.args = args
        self.label_model = label_model

    def extract_room_contours(self, img):
        kernel = self.args.contour_blur_kernel
        thr = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        contours, hierarchy = cv2.findContours(cv2.GaussianBlur(thr, (kernel, kernel), 0), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        return contours, hierarchy

    def extract_building_outline(self, img):
        kernel = 31

        dilation_shape = cv2.MORPH_RECT
        dilatation_size = self.args.outline_dilation
        element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),(dilatation_size, dilatation_size))

        thr = cv2.threshold(cv2.GaussianBlur(img, (kernel, kernel), 0), 180, 255, cv2.THRESH_BINARY_INV)[1]
        thr = cv2.dilate(thr, element)

        mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
        cv2.floodFill(thr, mask, [0, 0], [-1])
        mask = cv2.dilate(mask, element)[1:-1, 1:-1]

        contours_building, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        building_contour = sorted(contours_building, key=len)[-1]
        building_contour = cv2.approxPolyDP(building_contour, self.args.outline_eps * cv2.arcLength(building_contour, True), True)

        scale = [img.shape[1],img.shape[0]]

        return building_contour.reshape((building_contour.shape[0], 2)) / scale

    def extract_rooms(self, img, contours, hierarchy, out_patches = []):
        segmentation = []
        scale = [img.shape[1],img.shape[0]]

        for i, contour in enumerate(contours):
            first_child = hierarchy[0, i, 2]
            if i == 0 or first_child == -1:
                continue

            child = first_child
            count = 0
            while child != -1:
                count += 1
                child = hierarchy[0, child, 0]

            if count > 10:
                continue

            child = first_child

            room_num = -1

            while child != -1:  # Iterate over children
                (x,y,w,h) = cv2.boundingRect(contours[child])
                patch = img[y:y+h,x:x+w]
                label = self.label_model(patch)
                if 0 < len(label) < 7:
                    out_patches.append(patch)
                    room_num = label
                    break
                child = hierarchy[0, child, 0]

            if room_num != -1:
                contour_simp = cv2.approxPolyDP(contour, self.args.contour_eps * cv2.arcLength(contour, True), True)
                segmentation.append((room_num, contour_simp.reshape((contour_simp.shape[0], 2)) / scale))

        return segmentation

    def draw(self, result, segmentation):
        room_thick = 4
        building_thick = 10

        scale = np.array([result.shape[1],result.shape[0]])
        for (label, contour) in segmentation["locations"]:
            contour = (contour*scale).astype(np.int32)
            (x,y,w,h) = cv2.boundingRect(contour)
            cv2.putText(result, label, [x+w//3,y+h//2], cv2.FONT_HERSHEY_PLAIN, 2.0, [0.0, 0.0, 0.0, 1.0], 1)
            cv2.drawContours(result, [contour], 0, [0.0, 0.0, 0.0], room_thick)

        cv2.drawContours(result, [(segmentation["outline"]*scale).astype(np.int32)], 0, [0.0, 0.0, 0.0, 0.0], building_thick)

    def segment(self, img):
        contours, hierarchy = self.extract_room_contours(img)
        rooms = self.extract_rooms(img, contours, hierarchy)
        outline = self.extract_building_outline(img)
        return {"outline": outline, "locations": rooms}

    def apply_transform(self, matrix, segmentation):
        def apply_transform_to_contour(contour):
            homo = np.zeros((contour.shape[0], 3)) # euclidean -> homogenous
            homo[:,:2] = contour
            homo[:,2] = 1
            return np.einsum('ij,kj->ki', matrix, homo)[:,:2] # homogenous -> euclidean

        return {"outline": apply_transform_to_contour(segmentation["outline"]),
                "locations": [(label,apply_transform_to_contour(contour)) for label,contour in segmentation["locations"]]}

    def save(self, dst, img, segmentation):
        img_draw = 255*np.ones((img.shape[0],img.shape[1],3), dtype=img.dtype) #cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        self.draw(img_draw, segmentation)

        cv2.imwrite(dst+".png", img_draw)
        with open(dst+".json","w") as f:
            data = {
                "locations": [(room,contour.tolist()) for room,contour in segmentation["locations"]],
                "outline": [x.tolist() for x in segmentation["outline"]]
            }

            json.dump(data, f)


from . import digit_model, label_model, utils

with open(utils.DATA_FOLDER + "/default_config.yaml") as f:
    default_config = yaml.safe_load(f.read())

def segmentation_model(device, config=default_config):
    label_args = label_model.Label_Model_Args(**config["label"])
    segmentation_args = SegmentationArgs(**config["segmentation"])

    digit_m = digit_model.load_model(device)
    label_m = label_model.Label_Model(device, digit_m, label_args)
    segmentation_m = Segmentation_Model(label_m, segmentation_args)

    return segmentation_m

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dst")
    parser.add_argument("src")
    parser.add_argument("config", required=False)
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f.read())
    else:
        config = default_config

    device = utils.select_device()
    segmentation_m = segmentation_model(device, config)

    img = cv2.imread(args.src, cv2.IMREAD_GRAYSCALE)
    rooms = segmentation_m.segment(img)
    segmentation_m.save(args.dst, img, rooms)

