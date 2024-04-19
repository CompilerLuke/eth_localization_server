CREATE TABLE BuildingScan.FileUpload (
    id INT IDENTITY(1,1) PRIMARY KEY,
    building INT FOREIGN KEY REFERENCES BuildingModel.Buildings(id),
    username VARCHAR(128) NOT NULL,
    filename VARCHAR(256) NOT NULL,
    blobFilename VARCHAR(512),
    fileType VARCHAR(256) NOT NULL,
    metadata NVARCHAR(MAX)
);

ALTER TABLE BuildingScan.FileUpload
    ADD CONSTRAINT uq_filename UNIQUE(username,[filename]);