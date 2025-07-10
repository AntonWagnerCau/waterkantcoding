#!/bin/bash
# Navigate to the directory containing the docker-compose.yml
cd "$(dirname "$0")"
echo "Building and starting services..."
docker compose up --build -d
echo "Services started."
docker ps
