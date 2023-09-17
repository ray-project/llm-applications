#!/bin/bash
# Install postgres
sudo apt install -y wget ca-certificates
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt/ $(lsb_release -cs)-pgdg main" >> /etc/apt/sources.list.d/pgdg.list'
sudo apt update && sudo apt install -y postgresql postgresql-contrib
# Install pgvector
sudo apt install -y postgresql-server-dev-16
pushd /tmp && git clone --branch v0.4.4 https://github.com/pgvector/pgvector.git && pushd pgvector && make && sudo make install && popd && popd
# Activate pgvector and the database
echo 'ray ALL=(ALL:ALL) NOPASSWD:ALL' | sudo tee /etc/sudoers
sudo service postgresql start
# pragma: allowlist nextline secret
sudo -u postgres psql -c "ALTER USER postgres with password 'postgres';"
sudo -u postgres psql -c "CREATE EXTENSION vector;"
