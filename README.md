1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
    ```
   
2. **Create a virtual environment:**:
   ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
   
3. **Install dependencies:**:
   ```bash
    pip install -r requirements.txt

4. **Build and start database**:
   ```bash
   docker compose build db
   docker compose up -d db
5. **Apply migrations to db**:
   ```bash
   make docker-migrate-db

### Running the Application

1. **Run the FastAPI application:**:
   ```bash
   uvicorn main:app --reload
    ```
   
2. **Access the API documentation:**
   Open your browser and navigate to http://127.0.0.1:8000/docs to see the interactive API documentation.
