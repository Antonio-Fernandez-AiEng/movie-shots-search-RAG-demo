from fastapi.testclient import TestClient
from unittest.mock import patch
from main import app

# Create a client for virtual tests for the FastAPI
client = TestClient(app)

def test_health_check():
    """Tests that the enpoint works"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"
    print("✅ [TEST] Health check pasado con éxito.")


@patch("main.processor")
@patch("main.calculate_file_hash")
@patch("main.get_video_by_hash")
@patch("main.save_processed_video")
def test_ingest_video_endpoint_mocked(
    mock_save_video, mock_get_hash, mock_calc_hash, mock_processor
):
    """Tests the ingest endpoint (api/ingest) using Mocks to isolate the IA 
    and the anti-duplication logic for SQLite
    """
    
    # 1. Emulating the hashins and the cheking for duplicates on the SQL database
    mock_calc_hash.return_value = "fake_sha256_hash_12345"
    mock_get_hash.return_value = None  # Simulate the video doesnt exist

    # 2. Simulate the video processor for the background task
    mock_processor.extract_and_vectorize.return_value = [
        {
            "id": "test-shot-uuid-1",
            "metadata": {
                "movie_title": "Test Movie",
                "timestamp": 12.5,
                "file_path": "data/extracted_shots/test_shot.jpg",
                "shot_index": 0
            }
        }
    ]

    # 3. Simulate a false binary file (.mp4)
    fake_video_bytes = b"fake video content bytes for testing"
    files = {"files": ("test_trailer.mp4", fake_video_bytes, "video/mp4")}
    data = {"movie_title": ["Test Movie"]}

    # 4. Virtual POST petition to the endpoint
    response = client.post("/api/ingest", files=files, data=data)

    # 5. Verifications
    assert response.status_code == 202
    json_data = response.json()
    
    # Asses if we get the answer we expected with the right format
    assert json_data["status"] == "accepted"
    assert "task_id" in json_data
    assert json_data["movie_title"] == "Test Movie"
    assert "status_url" in json_data
    
    print("✅ [TEST] Ingest endpoint with duplicate protection verified correctly with mocks")


@patch("main.processor")
@patch("main.calculate_file_hash")
@patch("main.get_video_by_hash")
def test_ingest_duplicate_video_blocked(mock_get_hash, mock_calc_hash, mock_processor):
    """Tests if the system succesfully rejects a duplicate uploaded file (400 Bad Request)
    thanks to the hashes stored in the SQL database
    """
    mock_calc_hash.return_value = "existing_sha256_hash_999"
    # Simulate the SQL database finds the video already exists (task_id, movie_title)
    mock_get_hash.return_value = ("task-id-previo-abc", "Película Duplicada")

    fake_video_bytes = b"same video content bytes"
    files = {"files": ("duplicate_trailer.mp4", fake_video_bytes, "video/mp4")}

    response = client.post("/api/ingest", files=files)

    # Must give a 400 ERROR due to duplicates
    assert response.status_code == 400
    assert "ya fue procesado" in response.json()["detail"]
    print("✅ [TEST] Duplicate video blocking verified")

def test_get_task_status_not_found():
    """Tests if consulting a non-existent task id returns a 404 error"""
    fake_task_id = "task-que-no-existe-12345"
    response = client.get(f"/api/tasks/{fake_task_id}")
    
    assert response.status_code == 404
    assert response.json()["detail"] == "Task not found"
    print("✅ [TEST] Error handling for non-existents tasks verified")