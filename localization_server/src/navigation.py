import rustworkx
import json
import numpy as np
from dataclasses import dataclass
from building_model import BuildingModel
from matplotlib import pyplot as plt

with open("../data/outputs/CAB_navviz/segmentation/building.json") as f:
    building = BuildingModel.from_json(json.load(f))
    print("Loaded building", building)

def inside_contour(contour, pos):
    x,y = pos
    inside = False
    n = contour.shape[0]
    for i in range(n):
        xi,yi = contour[i%n]
        xj,yj = contour[(i+1)%n]
        intersect = (yi > y) != (yj > y) and (x < ((xj - xi) * (y - yi) / (yj - yi) + xi))
        if intersect:
            inside = not inside
    return inside


def nearest_node(building, pos):
    graph : rustworkx.PyGraph = building.graph
    room = None

    for room in building.floors[0].locations.values():
        if inside_contour(room.contour, pos[:2]):
            room = room.label
    if room:
        print("Found room", room)
        for id, node in zip(graph.node_indices(), graph.nodes()):
            if node.label == room:
                return id, node

    dist = np.array([np.linalg.norm(node.pos - pos) for node in graph.nodes()])
    return np.argmin(dist)

def navigate(building, src, dst):
    graph = building.graph

    if type(src) is str:
        src = graph.attrs["names_to_vertex"][src]
    if type(dst) is str:
        dst = graph.attrs["names_to_vertex"][dst]

    paths = rustworkx.dijkstra_shortest_paths(graph, src, dst, weight_fn=lambda x: x.length)
    nodes = []
    for i in paths[dst]:
        nodes.append(graph[i])
    return nodes


if __name__ == "__main__":
    path = navigate(building, "A", "D")
    print(nearest_node(building, [32.3, -37.4, 0]))
    print([p.to_json() for p in path])