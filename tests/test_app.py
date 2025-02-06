import pytest
import json
from app import create_app
from app.database import Base, engine
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Use the TestConfig from our config module
from app.config import TestConfig

# Override the engine for tests. Because our database.py uses engine at import time,
# one approach is to re-create the tables in the test database.
# For tests, we use an in-memory SQLite database.
@pytest.fixture(scope="session")
def test_engine():
    test_engine = create_engine(TestConfig.SQLALCHEMY_DATABASE_URI)
    Base.metadata.create_all(bind=test_engine)
    yield test_engine
    Base.metadata.drop_all(bind=test_engine)

# Create the Flask app using TestConfig
@pytest.fixture
def app():
    app = create_app(config_class=TestConfig)
    # Note: since our app's init_db() was already called, we assume the tables are created.
    # If necessary, you could call init_db() again here.
    yield app

# Create a test client for the app
@pytest.fixture
def client(app):
    return app.test_client()

# Example test for the index route
def test_index(client):
    response = client.get("/users/")
    assert response.status_code == 200
    assert b"Hello, Flask with PostgreSQL and SQLAlchemy!" in response.data

# Test creating a user via the POST /users endpoint
def test_create_user(client):
    payload = {"username": "testuser", "email": "testuser@example.com"}
    response = client.post("/users/add", json=payload)
    assert response.status_code == 201
    data = response.get_json()
    assert "id" in data
    assert data["username"] == "testuser"

# Test creating a duplicate user
def test_duplicate_user(client):
    payload = {"username": "duplicate", "email": "duplicate@example.com"}
    # Create the user the first time
    response1 = client.post("/users/add", json=payload)
    assert response1.status_code == 201
    
    # Try to create the same user again
    response2 = client.post("/users/add", json=payload)
    assert response2.status_code == 409

# Test listing users from the GET /users endpoint
def test_list_users(client):
    # Create two users
    client.post("/users/add", json={"username": "user1", "email": "user1@example.com"})
    client.post("/users/add", json={"username": "user2", "email": "user2@example.com"})
    
    # Now retrieve the list of users
    response = client.get("/users/list")
    assert response.status_code == 200
    data = response.get_json()
    # We expect at least 2 users in the response
    assert isinstance(data, list)
    assert len(data) >= 2
