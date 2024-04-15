from floor_plan_segmentation import segmentation as floor_segmentation, utils
import yaml
import cv2
from matplotlib import pyplot as plt

device = utils.select_device()
segmentation_m = floor_segmentation.segmentation_model(device)

def annotate(img):
    rooms = segmentation_m.segment(img)
    return rooms

if __name__ == "__main__":
    img = cv2.imread("../../floor_plan_segmentation/data/cab_floor_0.png", cv2.IMREAD_GRAYSCALE)
    rooms = annotate(img)

    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    segmentation_m.draw(img_color, rooms)
    plt.imshow(img_color)
    plt.show()