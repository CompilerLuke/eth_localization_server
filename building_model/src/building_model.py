from dataclasses import *
import numpy as np
import json
import rustworkx
from typing import List, Dict, Tuple
from enum import Enum
import pyodbc
import copy
import re

class LocationType(Enum):
    ROOM = 1
    CORRIDOR = 2


@dataclass
class NodeData:
    type: LocationType
    label: str
    pos: np.ndarray
    locationID: int

    def to_json(data):
        return {
            "type": data.type.name,
            "label": data.label,
            "pos": list(data.pos),
            "locationID": data.locationID,
        }

    @staticmethod
    def from_json(data):
        return NodeData(
                type= LocationType[data["type"]] if "type" in data else LocationType.ROOM,
                label=data["label"],
                pos=np.array(data["pos"]),
                locationID=int(data["locationID"])
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

# todo: add floor relation
@dataclass 
class Location:
    id: str
    label: str 
    type: str
    desc: str
    contour: np.ndarray 
    parent: int

    def to_json(self):
        return {
            "id": self.id,
            "label": self.label,
            "desc": self.desc,
            "type": self.type.value,
            "contour": list(list(p) for p in self.contour),
            "parent": self.parent
        }

@dataclass
class Room:
    id: int
    label: str
    type: str
    desc: str
    contour: np.ndarray

    def to_json(self):
        return {
            "id": self.id,
            "label": self.label,
            "type": self.type,
            "desc": self.desc,
            "contour": list(list(p) for p in self.contour)
        }

    @staticmethod
    def from_json(json):
        return Room(
            id=json["id"],
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
    locations: List[Room]
    outline: np.ndarray # (n,2)
    label: str
    num: int
    walkable_areas: List[np.ndarray] # (m,*,2)

    def to_json(self):
        return {
            "z": self.z,
            "min": list(self.min),
            "max": list(self.max),
            "locations": [room.to_json() for room in self.locations],
            "outline": [x.tolist() for x in self.outline],
            "label": self.label,
            "num": self.num,
            "walkable_areas": [x.tolist() for x in self.walkable_areas]
        }

    @staticmethod
    def from_json(json):
        return Floor(
            z= json["z"],
            min= json["min"],
            max= json["max"],
            locations= [Room.from_json(room) for room in json["locations"]],
            outline= np.array(json["outline"]),
            label=json["label"],
            num=json["num"],
            walkable_areas=[np.array(x) for x in json["walkable_areas"]]
        )


class BuildingModel:
    id: int
    name: str
    creator: str
    locations: Dict[int, Room]
    floors: List[Floor]
    graph: rustworkx.PyGraph

    def __init__(self, id: int, name: str, floors: List[Floor], graph: rustworkx.PyGraph, creator: str = ""):
        self.id = id
        self.name = name
        self.floors = floors
        self.graph = graph
        self.creator = creator

        self.locations = {}
        for floor in floors:
            for location in floor.locations:
                self.locations[location.id] = location

    @staticmethod
    def graph_to_json(graph: rustworkx.PyGraph):
        nodes = {id: data.to_json() for id, data in zip(graph.node_indices(), graph.nodes())}

        edges = {}
        for edge, data in enumerate(graph.edges()):
            u, v = graph.get_edge_endpoints_by_index(edge)
            edges[edge] = {
                "u": u,
                "v": v,
                "data": data.to_json(),
            }

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
        locations = {name: room.to_json() for name,room in self.locations.items()}
        floors = [floor.to_json() for floor in self.floors]
        graph = BuildingModel.graph_to_json(self.graph)

        return {
            "id": self.id,
            "name": self.name,
            "locations": locations,
            "floors": floors,
            "graph": graph
        }

    @staticmethod
    def from_json(json):
        floors = [Floor.from_json(floor) for floor in json["floors"]]
        graph = BuildingModel.graph_from_json(json["graph"])

        return BuildingModel(
            id=json["id"],
            name=json["name"],
            floors=floors,
            graph=graph
        )
    
def db_create_floor(cursor: pyodbc.Cursor, building_id: int, floors: List[Floor]):
    query = """
    INSERT INTO BuildingModel.Floors (label,floorNumber)
    VALUES (?, ?)
    """

    values = []
    for floor in floors:
        values.append((floor.label, floor.num))

    cursor.executemany(query, values)

def db_contour_to_polygon(contour):
    vals = [ "{} {}".format(x,y) for x,y in np.concatenate([contour,[contour[0]]])]
    contour_str = "POLYGON((" + ",".join(vals) + "))"
    return contour_str 

# todo: add floor
def db_create_rooms(cursor: pyodbc.Cursor, building_id: int, rooms: Dict[str, Room]):
    query1 = """
    INSERT INTO BuildingModel.Location (locationType, label, description, building, contour)
    OUTPUT inserted.id
    VALUES (?, ?, ?, ?, geometry::STPolyFromText(?,0));"""
        
    query2 = """
    INSERT INTO BuildingModel.Rooms (locationID, roomType)
    VALUES (?, ?)
    """

    values1 = []
    for _, room in rooms.items():
        values1.append((LocationType.ROOM.value, room.label, room.desc, building_id, db_contour_to_polygon(room.contour))) 
        
    cursor.executemany(query1, values1)
    ids = [row[0] for row in cursor.fetchall()]
    
    values2 = []
    for id,room in zip(ids, rooms.values()):
        values2.append((id, room.type))
    cursor.executemany(query2, values2)

    return {id: new_id for id, new_id in zip(rooms.keys(), ids)}

def db_create_nodes(cursor, graph, location_map = None):
    query = """
    INSERT INTO BuildingModel.NavigationVertices(position,locationID)
    OUTPUT inserted.ID
    VALUES (geometry::STGeomFromText(?,0),?);
    """

    values = []
    for node in graph.nodes():
        if node.locationID == -1:
            locationID = None
        elif location_map:
            locationID = location_map[node.locationID]
        else:
            locationID = node.locationID

        position = "Point("+" ".join(map(str, node.pos))+")"
        values.append((position, locationID,))

    node_map  = {}
    for id, val in enumerate(values): # workaround: fail to get fetchmany
        cursor.execute(query, val)
        node_map[id] = cursor.fetchone()[0]
    return node_map # {id: row[0] for id, row in zip(graph.node_indexes(), cursor.fetchall())}

def db_create_edges(cursor, graph, node_map):
    query = """
    INSERT INTO BuildingModel.NavigationEdges($from_id,$to_id,length)
    VALUES ((SELECT $node_id FROM BuildingModel.NavigationVertices WHERE id=?), 
            (SELECT $node_id FROM BuildingModel.NavigationVertices WHERE id=?), 
            ?)
    """

    def map_n(x):
        return node_map[x] if node_map else x
    
    values = []
    for i, edge in zip(graph.edge_indices(), graph.edges()):
        u, v = graph.get_edge_endpoints_by_index(i)
        if node_map:
            u = node_map[u]
            v = node_map[v]
        values.append((u,v,edge.length))
    cursor.executemany(query, values)

def db_create_building_entry(cursor: pyodbc.Cursor, building: BuildingModel):
    LONG_LAT = 4326 
    
    query = f"""
    INSERT INTO BuildingModel.Buildings (label,contour,[username])
    OUTPUT INSERTED.id 
    VALUES (?, geography::STPolyFromText(?,{LONG_LAT}), ?);
    """

    # todo: correct long/lat extent of building
    extent = [ 
        [45.5, 56.5],
        [56.5, 56.4],
        [30.4, 26.4]
    ]

    cursor.execute(query, (building.name, db_contour_to_polygon(extent), building.creator))
    new_id = cursor.fetchone()[0]
    return new_id

def db_create_building(conn: pyodbc.Connection, building: BuildingModel):
    cursor = conn.cursor()
    id = db_create_building_entry(cursor, building)
    cursor.commit()
    db_create_floor(cursor, id, building.floors)
    room_map = db_create_rooms(cursor, id, building.rooms)
    node_map = db_create_nodes(cursor, building.graph, room_map)
    db_create_edges(cursor, building.graph, node_map)

    cursor.commit()
    return id

def parse_geometry(geo):
    accum = ""
    lex = []
    for c in geo:
        if c in ['(',')',',',' ']:
            if accum:
                lex.append(accum)
            accum = ""
            if not c in ' ':
                lex.append(c)
        else:
            accum += c
    if accum:
        lex.append(accum)
    # print(geo, lex)

    if lex[0] == "POLYGON":
        if lex[1:2] == ['(','(']:
            return None
        start = 3
        end = lex.index(")")
        i = start
        points = []
        while True:
            x = float(lex[i])
            y = float(lex[i+1])
            points.append([x,y])
            i += 2
            if lex[i] != ",":
                break
            i += 1
        if lex[i] != ')':
            return None

        return np.array(points)
    else:
        return None

@dataclass
class LocationFilter:
    building_id: int
    matches: str = ""
    floor: int = None # todo: implement filter

def db_query_locations(cursor: pyodbc.Cursor, filter: LocationFilter):
    query= """
    SELECT id,locationType,label,description,contour.ToString(),parent 
    FROM BuildingModel.Location
    WHERE building=?
    """

    args = [filter.building_id]
    if filter.matches:
        query += "AND label LIKE ?"
        args.append(filter.matches+"%")

    query += ";"

    cursor.execute(query, args)

    result = []
    for id,locationType,label,desc,contour,parent in cursor.fetchall():
        result.append(Location(
            id=id,
            type=LocationType(locationType),
            label=label,
            desc=desc,
            contour=parse_geometry(contour),
            parent=parent
        ))

    print("Locations")
    print(result)
    return result

