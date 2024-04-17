from dataclasses import *
import numpy as np
import json
import rustworkx
from typing import List, Dict, Tuple
from enum import Enum


class NodeType(Enum):
    ROOM = 1
    CORRIDOR = 2


@dataclass
class NodeData:
    type: NodeType
    label: str
    pos: np.ndarray

    def to_json(data):
        return {
            "type": data.type.name,
            "label": data.label,
            "pos": list(data.pos)
        }

    @staticmethod
    def from_json(data):
        print(data["pos"])
        return NodeData(
                type= NodeType[data["type"]] if "type" in data else NodeType.ROOM,
                label=data["label"],
                pos=np.array(data["pos"])
            )

@dataclass
class EdgeData:
    length: float

    def to_json(self):
        return {
            "length": self.length
        }

    @staticmethod
    def from_json(data):
        return EdgeData(**data)


@dataclass
class Room:
    label: str
    type: str
    desc: str
    contour: np.ndarray

    def to_json(self):
        return {
            "label": self.label,
            "type": self.type,
            "desc": self.desc,
            "contour": list(list(p) for p in self.contour)
        }

    @staticmethod
    def from_json(json):
        return Room(
            label=json["label"],
            type=json["type"],
            desc=json["desc"],
            contour=np.array(json["contour"])
        )


@dataclass
class Floor:
    z: float # floor
    min: np.ndarray # (3,)
    max: np.ndarray # (3,)
    rooms: List[Room]
    outline: np.ndarray # (n,2)

    def to_json(self):
        return {
            "z": self.z,
            "min": list(self.min),
            "max": list(self.max),
            "rooms": [room.label for room in self.rooms],
            "outline": [x.tolist() for x in self.outline]
        }

    @staticmethod
    def from_json(rooms, json):
        return Floor(
            z= json["z"],
            min= json["min"],
            max= json["max"],
            rooms= [rooms[room] for room in json["rooms"]],
            outline= np.array(json["outline"])
        )


class BuildingModel:
    rooms: Dict[str, Room]
    floors: List[Floor]
    graph: rustworkx.PyGraph

    def __init__(self, rooms: Dict[str,Room], floors: List[Floor], graph: rustworkx.PyGraph):
        self.rooms = rooms
        self.floors = floors
        self.graph = graph

    @staticmethod
    def graph_to_json(graph):
        nodes = {id: data.to_json() for id, data in zip(graph.node_indexes(), graph.nodes())}

        edges = {}
        for edge, data in enumerate(graph.edges()):
            u, v = graph.get_edge_endpoints_by_index(edge)
            edges[edge] = {
                "u": u,
                "v": v,
                "data": data.to_json(),
            }

        print(list(graph.edge_indices()))
        return {"nodes": nodes, "edges": edges}

    @staticmethod
    def graph_from_json(json):
        idx_to_node = {}
        graph = rustworkx.PyGraph()

        names_to_vertex = {}
        for id, data in sorted(json["nodes"].items(), key=lambda x: x[0]):
            id = int(id)
            idx_to_node[id] = graph.add_node(NodeData.from_json(data))
            names_to_vertex[data["label"]] = id

        graph.attrs = {"names_to_vertex": names_to_vertex}

        for id, data in sorted(json["edges"].items(), key=lambda x: x[0]):
            graph.add_edge(idx_to_node[data["u"]], idx_to_node[data["v"]], EdgeData.from_json(data["data"]))

        return graph

    def to_json(self):
        rooms = {name: room.to_json() for name,room in self.rooms.items()}
        floors = [floor.to_json() for floor in self.floors]
        graph = BuildingModel.graph_to_json(self.graph)

        return {
            "rooms": rooms,
            "floors": floors,
            "graph": graph
        }

    @staticmethod
    def from_json(json):
        rooms = {name: Room.from_json(room) for name, room in json["rooms"].items()}
        floors = [Floor.from_json(rooms, floor) for floor in json["floors"]]
        graph = BuildingModel.graph_from_json(json["graph"])

        return BuildingModel(
            rooms=rooms,
            floors=floors,
            graph=graph
        )