 source ../env.sh
 docker run -e "ACCEPT_EULA=Y" -e "MSSQL_SA_PASSWORD=$SQL_SERVER_PASSWORD" \
   -p 1433:1433 --name localizr4 --hostname localizr \
   -d \
   mcr.microsoft.com/mssql/server:2022-latest