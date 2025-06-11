#!/bin/bash

# Wait for database
while ! nc -z db 5432; do
  sleep 1
done

# Generate Prisma client and push schema
prisma generate
prisma db push

# Start with hot reload (change this line based on your framework)
# For FastAPI:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# For Flask:
# export FLASK_ENV=development
# flask run --host 0.0.0.0 --port 8000

# For Django:
# python manage.py runserver 0.0.0.0:8000