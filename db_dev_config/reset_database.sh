source ../env.sh
sqlcmd -S localhost:1433 -U sa  -P $SQL_SERVER_PASSWORD -i drop_database.sql
sqlcmd -S localhost:1433 -U sa  -P $SQL_SERVER_PASSWORD -i create_database.sql
sqlcmd -S localhost:1433 -U sa  -P $SQL_SERVER_PASSWORD -d Localizr -i building_model/create_schema.sql
sqlcmd -S localhost:1433 -U sa  -P $SQL_SERVER_PASSWORD -d Localizr -i building_model/create_schema.sql
sqlcmd -S localhost:1433 -U sa  -P $SQL_SERVER_PASSWORD -d Localizr -i building_model/create_tables.sql
sqlcmd -S localhost:1433 -U sa  -P $SQL_SERVER_PASSWORD -d Localizr -i test.sql