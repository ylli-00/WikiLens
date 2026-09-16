#!/bin/bash
# Sourced by the mariadb image entrypoint once, when the data volume is first initialised.
# Creates a separate database for the test suite so tests never touch the ingested corpus.
docker_process_sql <<-SQL
	CREATE DATABASE IF NOT EXISTS \`${MARIADB_DATABASE}_test\`;
	GRANT ALL PRIVILEGES ON \`${MARIADB_DATABASE}_test\`.* TO '${MARIADB_USER}'@'%';
SQL
