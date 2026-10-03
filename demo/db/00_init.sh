#!/bin/sh
# Runs once, on an empty data volume, as the postgres superuser (docker-entrypoint-initdb.d).
# Only top-level files here are picked up by the entrypoint, so the SQL lives in sql/ and is
# run explicitly, with the application role's password passed as a psql variable.
set -eu

: "${ACL_APP_DB_PASSWORD:?ACL_APP_DB_PASSWORD must be set}"

psql --no-psqlrc -v ON_ERROR_STOP=1 \
    --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
    -v app_password="$ACL_APP_DB_PASSWORD" \
    -f /docker-entrypoint-initdb.d/sql/01_schema.sql \
    -f /docker-entrypoint-initdb.d/sql/02_seed.sql
