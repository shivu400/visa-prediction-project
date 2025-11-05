from fastapi.testclient import TestClient
import sys
import os

# Add the app directory to the path to allow importing 'main'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app')))
from main import app

client = TestClient(app)

def test_docs_exist():
    """
    Tests if the auto-generated docs page loads correctly.
    """
    response = client.get("/docs")
    assert response.status_code == 200