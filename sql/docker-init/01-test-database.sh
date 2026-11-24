#!/bin/bash
# Run by the mariadb image entrypoint once, when the data volume is first initialised.
# Creates a separate database for the test suite so tests never touch the ingested corpus.
# The entrypoint sources a *.sh file that is not executable and runs one that is as a program.
# A checkout on a file system without Unix file modes (a Windows drive under WSL) shows every
# file as executable, so the statements go through the entrypoint's docker_process_sql when it
# is defined (sourced) and through the mariadb client on the server's socket when it is not.
set -eo pipefail
test_db_sql="CREATE DATABASE IF NOT EXISTS \`${MARIADB_DATABASE}_test\`;
GRANT ALL PRIVILEGES ON \`${MARIADB_DATABASE}_test\`.* TO '${MARIADB_USER}'@'%';"
if declare -F docker_process_sql > /dev/null; then
	docker_process_sql <<< "$test_db_sql"
else
	MYSQL_PWD="$MARIADB_ROOT_PASSWORD" mariadb --protocol=socket -uroot -hlocalhost <<< "$test_db_sql"
fi
unset test_db_sql
