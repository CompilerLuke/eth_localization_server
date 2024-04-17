CREATE TABLE BuildingModel.Floors (
    id INT IDENTITY(1,1) PRIMARY KEY,
    label CHAR(128),
    floorNumber INT
);

CREATE TABLE BuildingModel.Buildings (
    id INT IDENTITY(1,1) PRIMARY KEY,
    label CHAR(128),
    contour GEOGRAPHY
);

CREATE TABLE BuildingModel.Location (
    id INT IDENTITY(1,1) PRIMARY KEY,
    locationType INT,
    label CHAR(128),
    contour GEOMETRY,
    building INT FOREIGN KEY REFERENCES BuildingModel.Buildings(id),
    parent INT FOREIGN KEY REFERENCES BuildingModel.Location(id)
);

CREATE TABLE BuildingModel.Rooms (
    id INT IDENTITY(1,1) PRIMARY KEY,
    locationID INT UNIQUE FOREIGN KEY REFERENCES BuildingModel.Location(id),
    roomType CHAR(128),
    description TEXT
);

CREATE TABLE BuildingModel.PointLandmarks (
    id INT IDENTITY(1,1) PRIMARY KEY,
    locationID INT UNIQUE FOREIGN KEY REFERENCES BuildingModel.Location(id),
    pointType CHAR(128),
    description TEXT
);

CREATE TABLE BuildingModel.Stairs (
    id INT IDENTITY(1,1) PRIMARY KEY,
    locationID INT UNIQUE FOREIGN KEY REFERENCES BuildingModel.Location(id)
);

CREATE TABLE BuildingModel.Corridors (
    id INT IDENTITY(1,1) PRIMARY KEY,
    locationID INT UNIQUE FOREIGN KEY REFERENCES BuildingModel.Location(id)
);

CREATE TABLE BuildingModel.NavigationVertices (
    id INT IDENTITY(1,1) PRIMARY KEY,
    position GEOMETRY,
    locationID INT FOREIGN KEY REFERENCES BuildingModel.Location(id)
) as NODE;

CREATE TABLE BuildingModel.NavigationEdges (
    length FLOAT
) as EDGE;

GO