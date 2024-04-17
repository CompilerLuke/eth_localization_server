from building_model import *
import pyodbc
import os

if __name__ == "__main__":
    with open("../../localization_server/data/outputs/CAB_navviz/segmentation/building.json", "r") as f:
        building = BuildingModel.from_json(json.load(f))

    host = os.environ["SQL_SERVER_HOST"]
    port = os.environ["SQL_SERVER_PORT"]  
    pwd = os.environ["SQL_SERVER_PASSWORD"]

    conn = pyodbc.connect("Driver={ODBC Driver 18 for SQL Server};"
                          +f"Server={host};"
                          +f"Port={port};"
                          +f"Database=master;"
                          +f"UID=sa;"
                          +f"PWD={pwd};"
                          +f"TrustServerCertificate=yes")

    db_create_building(conn, building)
