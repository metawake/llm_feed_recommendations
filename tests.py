###############################
# tests.py
###############################
import pytest
import asyncio
import os
from fastapi.testclient import TestClient
from httpx import AsyncClient
import sqlite3

# We'll import the 'app' from rss
from rss import app, DB_PATH, data_access, rss_service

# We need pytest-asyncio for async tests
pytestmark = pytest.mark.asyncio


@pytest.fixture
def client():
    """
    A standard synchronous TestClient for basic route testing.
    """
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session", autouse=True)
def setup_db():
    """
    Ensure a fresh database for tests, if desired.
    """
    # You might want to remove any existing test DB or use an in-memory DB:
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_preferences (
            user_id INTEGER PRIMARY KEY,
            topics TEXT
        )
    """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            link TEXT UNIQUE,
            published TEXT,
            content TEXT,
            summary TEXT
        )
    """
    )
    conn.close()
    yield


@pytest.mark.asyncio
async def test_set_user_preferences(client):
    """
    Tests setting user preferences via POST /users/{user_id}/preferences
    """
    response = client.post("/users/123/preferences", json={"topics": ["AI", "Python"]})
    assert response.status_code == 200
    assert response.json() == {"message": "Preferences saved."}

    row = data_access.fetch_one(
        "SELECT topics FROM user_preferences WHERE user_id = ?", (123,)
    )
    assert row is not None
    assert row["topics"] == "AI,Python"


@pytest.mark.asyncio
async def test_manual_rss_refresh(client):
    """
    Tests calling /rss/refresh which triggers the background task to refresh feeds.
    We do not verify the feed parsing result here (that would require real or mocked data).
    """
    response = client.post("/rss/refresh")
    assert response.status_code == 200
    assert response.json() == {
        "message": "RSS feeds are being refreshed in the background."
    }


@pytest.mark.asyncio
async def test_recommendations_workflow(client):
    """
    End-to-end check:
    1. Set user preferences
    2. Simulate a feed refresh (or call refresh_feeds() directly)
    3. Check that /users/{id}/recommendations returns results
    NOTE: This relies on real feeds & real OpenAI calls if not mocked.
    """

    # 1. Set Preferences
    client.post("/users/1/preferences", json={"topics": ["AI", "Technology"]})

    # 2. Manually call the refresh in a synchronous request
    #    If you want to REALLY parse the feed, we might do an async call:
    await rss_service.refresh_feeds()
    # This ensures articles are inserted & summarized in DB.

    # 3. Get recommendations
    #    Possibly we have no guaranteed matches unless the feed has AI/Technology articles.
    #    We'll just verify that it returns a 200 response & 'recommendations' in JSON.
    response = await _async_post(client, "/users/1/recommendations", json={})
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data

    # If there's relevant content, you'll see a list of articles with summary
    # For a real test, you'd want to confirm length or certain keys.


@pytest.mark.asyncio
async def test_agent_query(client):
    """
    Test the /assistant/query endpoint.
    This will do real similarity searches & real LLM calls if not mocked.
    """
    response = await _async_post(
        client, "/assistant/query", json={"question": "What's new in AI?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "response" in data


########################################################
# Helper function to do async post with TestClient
########################################################
async def _async_post(client, url, json):
    """
    Because TestClient is synchronous, we run it in a thread.
    """
    loop = asyncio.get_running_loop()

    def do_request():
        return client.post(url, json=json)

    return await loop.run_in_executor(None, do_request)
