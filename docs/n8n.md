# n8n Setup Guide

Deploy FlowDB on Digital Ocean with your n8n for free and easy data storage and vectorization.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Deployment](#deployment)
- [Connecting n8n to FlowDB](#connecting-n8n-to-flowdb)
- [Using FlowDB in n8n](#using-flowdb-in-n8n)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Digital Ocean Droplet with n8n already running via Docker
- SSH access to your droplet (using the console from the dashboard is simplest)
- FlowDB repository: [https://github.com/RecursionAI/FlowDB](https://github.com/RecursionAI/FlowDB)

## Deployment

### 1. SSH into your Digital Ocean droplet

Using console in the dashboard is easiest here. Just log in, go to your droplet, and click on the console

### 2. Clone FlowDB repository

```bash
cd /opt  # or wherever you want to store it
git clone https://github.com/RecursionAI/FlowDB.git
cd FlowDB
```

### 3. Configure your API key

Create your `.env` file:

```bash
touch .env
nano .env
```

#### Example `.env` file:

```ini
OPENAI_API_KEY = sk-proj-12345...
FLOWDB_API_KEY = my-super-secret-key
FLOWDB_PATH = ./my_db_storage
FLOWDB_VECTORIZER = openai
```

### 4. Start FlowDB

```bash
docker compose up -d
```

### 5. Verify FlowDB is running

```bash
docker ps
# Look for a FlowDB container
```

## Connecting n8n to FlowDB

### 1. Check container names

```bash
docker ps
```

You should see:

- `flowdb_server` (or similar)
- `n8n-docker-caddy-n8n` (or your n8n container name)

### 2. Check which networks they're on

```bash
docker inspect flowdb_server | grep -A 10 Networks
docker inspect n8n-docker-caddy-n8n | grep -A 10 Networks
```

### 3. Connect FlowDB to n8n's network

```bash
# Replace with your actual n8n network name (likely n8n-docker-caddy_default)
docker network connect n8n-docker-caddy_default flowdb_server
```

## Using FlowDB in n8n

Now that it's running you can start using FlowDB in n8n via the HTTP Request Node!

Here's a **comprehensive** guide on how you can use it.

### Schemas:

If you configured an API key you must always include an Authorization header in your requests with the value
`Bearer your-api-key`.

Look at the [Schemas](schemas.md) page for details on API schemas and url configurations.

### Creating/Updating Data (Upsert)

**Add an HTTP Request node in n8n:**/

- **Method:** `POST`
- **URL:** `http://flowdb_server:8000/v1/users/upsert`
- **Authentication:** Header Auth
    - Name: `Authorization`
    - Value: `Bearer your-api-key`
- **Body Content Type:** JSON
- **Body:**

```json
{
  "id": "unique-id-here",
  "data": {
    "name": "Example",
    "category": "AI Researcher",
    "content": "Your content here"
  }
}
```

**Important:** The `data` field must be a **single object**, not an array.

### Retrieving All Data

**HTTP Request node:**

- **Method:** `GET`
- **URL:** `http://flowdb_server:8000/v1/users`
- **Authentication:** Header Auth (same as above)
- **No body needed**

### Searching/Filtering Data

**HTTP Request node:**

- **Method:** `POST`
- **URL:** `http://flowdb_server:8000/v1/users/search`
- **Authentication:** Header Auth
- **Body:**

```json
{
  "query": {
    "category": "AI Researcher"
  }
}
```

### Example: Storing Multiple Items

If you have an array of items from a previous node, use a **Loop Over Items** or **Split Out** node first:

```
Your Data (array) 
  ↓
Split Out (on the array field)
  ↓
HTTP Request (Upsert)
  ↓
Each item saved individually
```

**In the HTTP Request body:**

```json
{
  "id": "{{ $json.id }}",
  "data": {{
  $json
}}
}
```

## Common Patterns

### Pattern 1: Store scraped data daily

```
Schedule Trigger (daily)
  ↓
Web Scraper
  ↓
Code: Add date and format
  ↓
HTTP Request: Upsert to FlowDB
```

### Pattern 2: Retrieve filtered content

```
Manual Trigger
  ↓
HTTP Request: Search FlowDB
  Body: { "query": { "category": "AI Researcher" } }
  ↓
Use results in next node
```

### Pattern 3: Newsletter workflow

```
Google Sheets (get subscribers)
  ↓
Loop Over Items (each subscriber)
  ↓
HTTP Request: Search FlowDB (filtered by their preferences)
  ↓
AI: Generate personalized email
  ↓
Gmail: Send email
```

## Troubleshooting

### Error: "ECONNREFUSED"

**Problem:** n8n can't reach FlowDB

**Solution:**

1. Verify both containers are running: `docker ps`
2. Check they're on the same network (see [Connecting n8n to FlowDB](#connecting-n8n-to-flowdb))
3. Use container name in URL: `http://flowdb:8000` or `http://flowdb_server:8000`

### Error: "Input should be a valid dictionary"

**Problem:** You're sending an array to the `data` field

**Solution:** FlowDB expects `data` to be a single object. If you have multiple items:

- Use Loop Over Items or Split Out to process one at a time
- Or wrap your array: `{ "id": "...", "data": { "items": [your array] } }`

### Error: "Input should be a valid string" (for id field)

**Problem:** The `id` field is receiving a number instead of a string

**Solution:** Convert to string: `"id": "{{ $json.id.toString() }}"` or use a string value directly

### FlowDB container keeps restarting

**Check logs:**

```bash
docker logs flowdb_server
```

Common issues:

- Missing API key in environment variables
- Port 8000 already in use
- Invalid docker-compose configuration

## File Locations

- **FlowDB installation:** `/opt/FlowDB` (or wherever you cloned it)
- **Data persistence:** Check your docker-compose.yml for volume mounts
- **Logs:** `docker logs flowdb_server`

## API Reference

For complete FlowDB API documentation, see: [FlowDB GitHub](https://github.com/RecursionAI/FlowDB)

## Related Documentation

- [n8n Official Docs](https://docs.n8n.io/)
- [Docker Networking](https://docs.docker.com/network/)
- [FlowDB Repository](https://github.com/RecursionAI/FlowDB)