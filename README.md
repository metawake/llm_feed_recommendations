
This project is a reference implementation that builds RSS feed recommendations service based on
FastAPI, LangChain, ChromaDB as vector database and ChatGPT as LLM provider.

To run a project you need to fill in OPENAPI key in .env file, then build and run docker:

docker build -t rss_aggregator_ai .
docker run -p 8000:8000 --name rss_ai_container rss_aggregator_ai

Once Docker container starts, it fills in initial feeds and creates vectors (see container log output).
After that use postman collection file to make API calls.

Specifically, that you can call /user_preferences endpoint to set preferences, and then
invoke /recommendations endpoint to see the list of articles recommended by topic.


A postman collection is in repo.
Or just use these calls:

curl --location 'http://localhost:8000/users/1/preferences' \
--header 'Content-Type: application/json' \
--data '{
    "topics": ["blockchain"]
}'

curl --location --request POST 'http://localhost:8000/users/1/recommendations' \
--header 'Content-Type: application/json' \
--data ''