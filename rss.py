import ast
import os
import json
import sqlite3
import feedparser
from typing import List, Dict

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi_utils.tasks import repeat_every
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# Logging
import logging
from pythonjsonlogger import jsonlogger

# Concurrency/async
import asyncio
import anyio
from anyio import to_thread

# LangChain / OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI


########################################
# Load environment variables
########################################
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_PATH = os.getenv("DB_PATH", "rss_ai.db")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./chroma_db")

# prevent from loading too much
MAX_FEEDS = int(os.getenv("MAX_FEEDS", "5"))
MAX_FEED_ITEMS = int(os.getenv("MAX_FEED_ITEMS", "10"))
MAX_RECOMMENDATIONS = int(os.getenv("MAX_RECOMMENDATIONS", "5"))

# Parse a JSON-encoded list of feeds from .env
DEFAULT_FEEDS_ENV = os.getenv("DEFAULT_FEEDS", "[]")
try:
    DEFAULT_FEEDS = ast.literal_eval(DEFAULT_FEEDS_ENV)

except json.JSONDecodeError as e:
    print(e)
    DEFAULT_FEEDS = []


########################################
# Initialize JSON logger
########################################
logger = logging.getLogger("rss_ai_logger")
logger.setLevel(logging.INFO)

log_handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
log_handler.setFormatter(formatter)
logger.addHandler(log_handler)

logger.info(
    "Starting RSS AI Service",
    extra={"db_path": DB_PATH, "vector_store": VECTOR_STORE_PATH},
)


########################################
# FastAPI initialization
########################################
app = FastAPI(title="RSS Aggregator with AI - Async Summaries")


########################################
# Pydantic Models
########################################
class UserPreferences(BaseModel):
    topics: List[str]


class RecommendationResponse(BaseModel):
    recommendations: List[Dict[str, str]]


class AgentResponse(BaseModel):
    response: List[Dict[str, str]]


class AgentQuery(BaseModel):
    question: str


########################################
# Custom Exceptions
########################################
class UserNotFoundException(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=404, detail="User not found or preferences not set."
        )


class RSSFeedException(HTTPException):
    def __init__(self):
        super().__init__(status_code=500, detail="Error processing RSS feeds.")


########################################
# Data Access
########################################
class SQLDataAccess:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def execute_query(self, query: str, params: tuple = ()):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        conn.close()

    def fetch_one(self, query: str, params: tuple = ()):
        conn = self.get_connection()
        cursor = conn.cursor()
        result = cursor.execute(query, params).fetchone()
        conn.close()
        return result

    def fetch_all(self, query: str, params: tuple = ()):
        conn = self.get_connection()
        cursor = conn.cursor()
        results = cursor.execute(query, params).fetchall()
        conn.close()
        return results


class VectorDataAccess:
    def __init__(self, persist_directory: str):
        self.persist_directory = persist_directory
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
        )

    def add_texts(self, texts: List[str], metadatas: List[dict]):
        self.vector_store.add_texts(texts, metadatas=metadatas)

    def similarity_search(self, query: str, k: int = 5):
        return self.vector_store.similarity_search(query, k=k)

    def persist(self):
        self.vector_store.persist()


########################################
# Services
########################################
class RSSService:
    def __init__(
        self, sql_data_access: SQLDataAccess, vector_data_access: VectorDataAccess
    ):
        self.sql_data_access = sql_data_access
        self.vector_data_access = vector_data_access
        self.rss_feeds = DEFAULT_FEEDS or [
            "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml"
        ]
        self.rss_feeds = self.rss_feeds[:MAX_FEEDS]

    async def refresh_feeds(self):
        """
        Refresh all RSS feeds asynchronously.
        Each feed is processed in parallel, and each article is summarized
        and inserted into the DB if not already present.
        """
        tasks = []
        logger.info("Starting feed refresh", extra={"feeds_count": len(self.rss_feeds)})
        for feed_url in self.rss_feeds:
            tasks.append(self.process_single_feed(feed_url))
        try:
            await asyncio.gather(*tasks)
        except Exception as exc:
            logger.error("Error refreshing feeds", extra={"error": str(exc)})
            raise RSSFeedException()
        logger.info("Finishing feed refresh")

    async def process_single_feed(self, feed_url: str):
        logger.info("Processing feed", extra={"feed_url": feed_url})
        feed = await to_thread.run_sync(feedparser.parse, feed_url)

        if not feed.entries:
            logger.warning(f"No entries found in feed: {feed_url}")
        else:
            logger.info(f"Processing {len(feed.entries)} entries for feed: {feed_url}")

        tasks = []
        entries = feed.entries[:MAX_FEED_ITEMS]
        for entry in entries:
            tasks.append(self.process_single_entry(entry))
        await asyncio.gather(*tasks)

    async def process_single_entry(self, entry):
        title = entry.get("title", "")
        link = entry.get("link", "")
        published = entry.get("published", "")
        content = entry.get("summary", "")

        if not title or not link:
            logger.warning(f"no title or link for {title} or {link}")
            return
        else:
            logger.warning(f"processing {title} or {link}")

        # Insert article ignoring duplicates
        self.sql_data_access.execute_query(
            """
            INSERT OR IGNORE INTO articles (title, link, published, content)
            VALUES (?, ?, ?, ?)
            """,
            (title, link, published, content),
        )

        # Check if summary is already stored
        row = self.sql_data_access.fetch_one(
            "SELECT summary FROM articles WHERE link = ?", (link,)
        )
        if row and row["summary"]:
            logger.info("Entry already summarized", extra={"link": link})
            return

        # Summarize using LLM call in a threadpool
        summary = await to_thread.run_sync(self.summarize_content, title, content)
        logger.info("Entry summarized", extra={"link": link})

        # Update DB with summary
        self.sql_data_access.execute_query(
            "UPDATE articles SET summary = ? WHERE link = ?", (summary, link)
        )

        self.vector_data_access.add_texts(
            [content + " " + summary], metadatas=[{"title": title, "link": link}]
        )

    def summarize_content(self, title: str, content: str) -> str:
        """Blocking call to LLM. We'll run it in a thread from async context."""
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
        prompt = f"Summarize this article:\nTitle: {title}\nContent: {content}"
        return llm.predict(prompt)


class UserService:
    def __init__(self, sql_data_access: SQLDataAccess):
        self.sql_data_access = sql_data_access

    def set_preferences(self, user_id: int, preferences: UserPreferences):
        logger.info(
            "Setting preferences",
            extra={"user_id": user_id, "topics": preferences.topics},
        )
        self.sql_data_access.execute_query(
            """
            INSERT OR REPLACE INTO user_preferences (user_id, topics)
            VALUES (?, ?)
            """,
            (user_id, ",".join(preferences.topics)),
        )


class RecommendationService:
    def __init__(
        self, sql_data_access: SQLDataAccess, vector_data_access: VectorDataAccess
    ):
        self.sql_data_access = sql_data_access
        self.vector_data_access = vector_data_access

    async def get_recommendations(self, user_id: int):
        logger.info("Getting recommendations", extra={"user_id": user_id})
        user = self.sql_data_access.fetch_one(
            "SELECT topics FROM user_preferences WHERE user_id = ?", (user_id,)
        )
        if not user:
            logger.warning("User not found", extra={"user_id": user_id})
            raise UserNotFoundException()

        topics = user["topics"].split(",")
        query = " OR ".join(topics)
        logger.info("Performing similarity search", extra={"query": query})
        results = self.vector_data_access.similarity_search(
            query, k=MAX_RECOMMENDATIONS
        )

        # Return the stored summary from the DB, rather than calling LLM again
        recommendations = []
        for result in results:
            link = result.metadata["link"]
            row = self.sql_data_access.fetch_one(
                "SELECT title, summary FROM articles WHERE link = ?", (link,)
            )
            if row:
                recommendations.append(
                    {
                        "title": row["title"],
                        "link": link,
                        "summary": row["summary"] or "",
                    }
                )

        return {"recommendations": recommendations}


class AgentService:
    def __init__(self, vector_data_access: VectorDataAccess):
        self.vector_data_access = vector_data_access

    async def query_agent(self, query: AgentQuery):
        """
        Agent approach that queries vector store by user question
        and fetches fresh LLM-based summary for each result
        (not stored in DB).
        """
        logger.info("Agent query", extra={"question": query.question})
        results = self.vector_data_access.similarity_search(query.question, k=5)

        tasks = []
        for r in results:
            tasks.append(self.summarize_agent(r))
        summaries = await asyncio.gather(*tasks)

        response_list = []
        for r, s in zip(results, summaries):
            response_list.append(
                {"title": r.metadata["title"], "link": r.metadata["link"], "summary": s}
            )
        return {"response": response_list}

    async def summarize_agent(self, result):
        """Async convenience to summarize with LLM in a threadpool."""
        return await to_thread.run_sync(self.agent_llm_predict, result)

    def agent_llm_predict(self, result):
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
        prompt = (
            f"Summarize this article: {result.metadata['title']}\n{result.page_content}"
        )
        return llm.predict(prompt)


########################################
# Instantiate data access and services
########################################
data_access = SQLDataAccess(DB_PATH)
vector_access = VectorDataAccess(VECTOR_STORE_PATH)

rss_service = RSSService(data_access, vector_access)
user_service = UserService(data_access)
recommendation_service = RecommendationService(data_access, vector_access)
agent_service = AgentService(vector_access)


########################################
# Setup DB at startup
########################################
@app.on_event("startup")
def init_db():
    """
    Ensure required tables exist in the database at startup.
    """
    create_user_preferences = """
    CREATE TABLE IF NOT EXISTS user_preferences (
        user_id INTEGER PRIMARY KEY,
        topics TEXT
    )
    """
    create_articles = """
    CREATE TABLE IF NOT EXISTS articles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        link TEXT UNIQUE,
        published TEXT,
        content TEXT,
        summary TEXT
    )
    """

    data_access.execute_query(create_user_preferences)
    data_access.execute_query(create_articles)
    logger.info("Database tables ensured/created")


########################################
# Periodic Refresh
########################################
@app.on_event("startup")
@repeat_every(seconds=1800)  # Every 30 minutes
async def refresh_rss_feeds():
    """Periodic refresh of RSS feeds (async)."""
    await rss_service.refresh_feeds()


########################################
# Routes
########################################
@app.post("/rss/refresh")
async def manual_refresh_rss_feeds(background_tasks: BackgroundTasks):
    """Manually refresh RSS feeds."""
    background_tasks.add_task(rss_service.refresh_feeds)
    return {"message": "RSS feeds are being refreshed in the background."}


@app.post("/users/{user_id}/preferences", response_model=Dict[str, str])
def set_user_preferences(user_id: int, preferences: UserPreferences):
    """Set user preferences for recommendations."""
    user_service.set_preferences(user_id, preferences)
    return {"message": "Preferences saved."}


@app.post("/users/{user_id}/recommendations", response_model=RecommendationResponse)
async def get_recommendations(user_id: int):
    """Get recommendations for a user based on their preferences."""
    return await recommendation_service.get_recommendations(user_id)


@app.post("/assistant/query", response_model=AgentResponse)
async def agent_query(query: AgentQuery):
    """Query the agent for recommendations or summaries based on a question."""
    return await agent_service.query_agent(query)


########################################
# Main Entry
########################################
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
