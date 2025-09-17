#!/bin/bash

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    echo "PORT=8080" > .env
    echo "ENVIRONMENT=development" >> .env
    echo "PYTHONPATH=/app" >> .env
fi  

# Build and start containers in detached mode
echo "Building and starting containers..."
docker compose up --build -d

# Follow logs of the main service
echo "Following container logs..."
docker compose logs -f