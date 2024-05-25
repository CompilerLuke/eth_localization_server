from floor_plan_segmentation import segmentation as floor_segmentation, utils
from building_model import BuildingModel, Room, Floor, LocationType
import json
import yaml
import cv2
from matplotlib import pyplot as plt
import numpy as np
from typing import Dict
import torch

device = utils.select_device()
segmentation_m = floor_segmentation.segmentation_model(device)

import localization_server

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

    for room in floor.locations:
        xc,yc = plot_contour(room.contour)
        xs += xc
        ys += yc

    trajectories = localization_server.localize.parse_trajectories("../../localization_server/data/HG_navviz/trajectories.txt")
    poses = []
    for i in range(500):
        idx = str(i).zfill(5) + "-cam0_center"
        final_pose = trajectories[idx].t
        poses.append(final_pose)
    poses = torch.stack(poses)

    #loc_sess = [14.6667, -12.9764,   0.9652] #[-13.8121, 12.7905, 0.9935]

    xo,yo = plot_contour(floor.outline)
    ax.plot(xs, ys)
    #ax.plot(xo, yo)
    ax.scatter(poses[:,0], poses[:,1], color="red")
    ax.set_aspect('equal', adjustable='box')

if __name__ == "__main__":
    building_path = "../../localization_server/data/HG_navviz"
    building_path_output = building_path + "/reference"

    img = cv2.imread(building_path + "/raw_data/floor_plan/hg_floor_plan_g.png", cv2.IMREAD_GRAYSCALE)
    segmentation = annotate(img)

    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    segmentation_m.draw(img_color, segmentation)
    segmentation_m.save(building_path_output+"/segmentation/hg_floor_plan_g_segmentation", img, segmentation)

    with open("../../localization_server/data/outputs/CAB_navviz/segmentation/graph.json") as f:
        graph = BuildingModel.graph_from_json(json.load(f))

    with open(building_path_output + "/transforms/transforms.json") as f:
        transforms = json.load(f)

    with open(building_path_output + "/segmentation/walkable_areas.json") as f:
        walkable_areas = json.load(f)["walkable_areas"]
        walkable_areas = [np.array(x) for x in walkable_areas]

    aabb_min = transforms["min"]
    aabb_max = transforms["max"]

    floor_to_relative = np.array(transforms["floor_to_relative"])
    relative_to_absolute = np.array(transforms["relative_to_absolute"])

    segmentation = segmentation_m.apply_transform(relative_to_absolute @ floor_to_relative, segmentation)

    locations = [Room(id=id, type="regular", label=label, desc="", contour=contour) for id, (label, contour) in enumerate(segmentation["locations"])]

    floor = Floor(
        label="E",
        num=0,
        outline=segmentation["outline"],
        locations=locations,
        z=aabb_min[2],
        min=aabb_min[:2],
        max=aabb_max[:2],
        walkable_areas=walkable_areas
    )

    building_model = BuildingModel(
        id=0,
        name="CAB",
        graph=graph,
        floors=[floor]
    )

    building_json_output = building_path_output + "/segmentation/building.json"
    with open(building_json_output, "w") as f:
        json.dump(building_model.to_json(), f)

    with open(building_json_output, "r") as f:
        BuildingModel.from_json(json.load(f))

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img_color)
    plot_floor(axes[1], floor)

    plt.show()
