import os
from fastapi.testclient import TestClient
from ..backend import app

os.environ["TESTING"] = "1"  # Set testing env before app startup

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to our API page!"}
