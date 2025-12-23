# URL Schemas:

FlowDB is API controlled. Here is how the URL schemas work:

-----

## Base URL:

First you need to get your base url. FlowDB defaults to port 8000.

If running locally your base url will look like `http://localhost:8000` or `http://0.0.0.0:8000`.

If running via docker in Digital ocean you need to find your port and optionally your port alias.

run `docker ps` to see your running containers.

Look at the ports for the flowdb service. You should see something like `0.0.0.0-flowdb_server`. You can use either
`0.0.0.0` or `flowdb_server` as your base url.
**Example Base URLs:**

- Local: `http://localhost:8000`
- Docker: `http://0.0.0.0:8000` or `http://flowdb_server:8000`

-----

## Auth

If you specified an API Key in your `.env` file you must path an Authorization header with the bearer token.

`{"Authorization": "Bearer {API_KEY}"}`

## API Endpoints

### Upsert

**POST**: `/v1/{collection_name}/upsert`

**JSON Body**:
```json
{
  "id": "1",
  "data": {}
}
```
Your objects must have an `id` field for mapping and `data` must be a valid JSON object.


### Read

**GET**: `/v1/{collection_name}/read/{id}`

### List (with pagination)

**GET**: `/v1/{collection_name}/list`

### Semantic Search (RAG)

**POST**: `/v1/{collection_name}/search`

**BODY**:
```json
{
  "query_text": "What is the capital of France?",
  "collection_name": "countries",
  "limit": 5,
  "top_k": 5
}
```

### Delete

**DELETE**: `/v1/{collection_name}/delete/{key}`


### List Collections

**GET**: `/v1/collections`
