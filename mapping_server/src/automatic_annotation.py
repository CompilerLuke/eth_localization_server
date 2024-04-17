from floor_plan_segmentation import segmentation as floor_segmentation, utils
from building_model import BuildingModel, Room, Floor, NodeType
import json
import yaml
import cv2
from matplotlib import pyplot as plt
import numpy as np
from typing import Dict

device = utils.select_device()
segmentation_m = floor_segmentation.segmentation_model(device)


def annotate(img):
    rooms = segmentation_m.segment(img)
    return rooms


def plot_floor(ax, floor):
    xs = []
    ys = []

    def plot_contour(contour):
        start = contour[0:-1, :]
        end = contour[1:, :]

        p = np.zeros((3 * (contour.shape[0] - 1), 2))
        p[0::3, :] = start
        p[1::3, :] = end
        p[2::3, :] = np.nan
        return list(p[:, 0]), list(p[:, 1])

    for id,room in floor.rooms.items():
        xc,yc = plot_contour(room.contour)
        xs += xc
        ys += yc

    xo,yo = plot_contour(floor.outline)
    ax.plot(xs, ys)
    ax.plot(xo, yo)
    ax.set_aspect('equal', adjustable='box')

if __name__ == "__main__":
    img = cv2.imread("../../floor_plan_segmentation/data/cab_floor_0.png", cv2.IMREAD_GRAYSCALE)
    segmentation = annotate(img)

    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    segmentation_m.draw(img_color, segmentation)
    segmentation_m.save("../../floor_plan_segmentation/data/cab_floor_segmentation", img, segmentation)

    with open("../../localization_server/data/outputs/CAB_navviz/segmentation/graph.json") as f:
        graph = BuildingModel.graph_from_json(json.load(f))

    aabb_min = np.array([-80.19033051, -160.14266968, -14.49797497])
    aabb_max = np.array([173.27059428, 107.40621185, 46.93898773])

    floor_to_relative = np.array([[0.17137878, 0.34187206, 0.21509406],
                                  [-0.38568048, 0.1352338, 0.70865127],
                                  [0., 0., 1.]])
    relative_to_absolute = np.array([[253.46092479, 0., -80.19033051],
                                     [   0.,        -267.54888153,  107.40621185],
                                     [0., 0., 1.]])

    floor_to_absolute = relative_to_absolute @ floor_to_relative

    segmentation = segmentation_m.apply_transform(floor_to_absolute, segmentation)

    rooms = {id: Room(type="regular", label=label, desc="", contour=contour) for id, (label, contour) in enumerate(segmentation["rooms"])}

    floor = Floor(
        label="E",
        num=0,
        outline=segmentation["outline"],
        rooms=rooms,
        z=aabb_min[2],
        min=aabb_min[:2],
        max=aabb_max[:2]
    )

    building_model = BuildingModel(
        id=0,
        name="CAB",
        rooms=rooms,
        graph=graph,
        floors=[floor]
    )

    with open("../../localization_server/data/outputs/CAB_navviz/segmentation/building.json", "w") as f:
        json.dump(building_model.to_json(), f)

    with open("../../localization_server/data/outputs/CAB_navviz/segmentation/building.json", "r") as f:
        BuildingModel.from_json(json.load(f))

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img_color)
    plot_floor(axes[1], floor)

    plt.show()
