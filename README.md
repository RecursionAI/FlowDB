# FlowDB

**The AI-Native Database for Agents, Automations, and Speed.**

FlowDB is a **Hybrid Vector-JSON Database** designed for the age of AI. It replaces the complex "Postgres + Pinecone + Docker + Glue Code" stack with a single, lightweight Python package.

It is **MCP-Native**, meaning LLMs (like Claude) can read, write, and search your database directly without you writing a single line of API wrapper code.

-----

## The Philosophy: "SQLite for Agents"

Traditional databases (SQL) are built for accounting. They require rigid schemas, migrations, and complex setup.
**FlowDB is built for Agents.** It assumes that:

1.  **Data is messy:** You want to store JSON documents, not rows.
2.  **Meaning matters:** Everything should be vector-searchable by default.
3.  **Speed is life:** You shouldn't need 3 containers just to save a user preference.

### ✅ When to use FlowDB

  * **AI Agents & Assistants:** Give Claude/OpenAI long-term memory in 5 minutes.
  * **Local-First Automation (n8n):** Pipe raw JSON from webhooks directly into storage without schema errors.
  * **Rapid Prototyping:** Build backend APIs in minutes using Pydantic models.
  * **RAG Applications:** "Chat with your data" apps where setup speed \> complex relations.

### ❌ When NOT to use FlowDB

  * **Fintech / High-Stakes Transactional Systems:** If you need complex ACID compliance across multiple tables (e.g., bank transfers), use Postgres.
  * **Massive Relational Data:** If you need 10-way `JOIN`s, use a SQL database.

-----

## Features

  * **JSON Native:** No schemas. No migrations. Just upsert Pydantic models or raw JSON.
  * **Auto-Vectorization:** Automatically embeds your data using OpenAI (or custom models) on write. No separate embedding pipeline required.
  * **Hybrid Architecture:** Runs as a local background process (for dev) or a Docker container (for production).
  * **MCP Ready:** Exposes a Model Context Protocol (MCP) interface, turning your database into a "Long Term Memory" tool for AI Agents or giving you AI assistance with your data.
  * **Fast:** Built on **LMDB** (Lightning Memory-Mapped Database) and **USearch** (HNSW Vector Search).

-----

## 📦 Installation

```bash
pip install flowdb
```

-----

## Configuration

FlowDB is configured via a `.env` file in your project root or via Environment Variables.

| Variable | Description | Required? | Default |
| :--- | :--- | :--- | :--- |
| `OPENAI_API_KEY` | Your OpenAI Key for auto-vectorization. | **Yes** | `None` |
| `FLOWDB_API_KEY` | Secures the DB. If set, clients must provide this key. | No | `None` (Open Access) |
| `FLOWDB_PATH` | Where data files are stored on disk. | No | `./flow_data` |
| `FLOWDB_VECTORIZER` | Provider to use (`openai` or `mock`). | No | `auto` |

**Example `.env` file:**

```ini
OPENAI_API_KEY=sk-proj-12345...
FLOWDB_API_KEY=my-super-secret-password
FLOWDB_PATH=./my_db_storage
FLOWDB_VECTORIZER=openai
```

-----

## Server Usage

You need a running server to store data. You have two ways to run it.

### Option A: The CLI (Local Dev)

The easiest way to start developing locally.

```bash
# Starts the API at http://localhost:8000
flowdb start
```

### Option B: Docker (Production)

The robust way to deploy to DigitalOcean, Railway, or AWS.

**`docker-compose.yml`**

```yaml
services:
  flowdb:
    image: python:3.11-slim
    # We use 'sh -c' so we can chain commands with &&
    command: >
      sh -c "pip install flowdb-ai && flowdb start --host 0.0.0.0"
    ports:
      - "8000:8000"
    volumes:
      - ./flow_data:/app/flow_data
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - FLOWDB_API_KEY=${FLOWDB_API_KEY}
```

-----

## Client SDK Usage

FlowDB comes with a Pythonic Client SDK. It feels like using a local dictionary, but it talks to your high-performance server.

### 1\. Connection

```python
from flowdb import FlowDB
from pydantic import BaseModel

# Connects to localhost:8000 by default
# Reads FLOWDB_API_KEY from env automatically if present
db = FlowDB()

# Or connect to a remote production server
# db = FlowDB(url="http://164.x.x.x:8000", api_key="secret")
```

### 2\. Define Data Model

FlowDB uses **Pydantic** to validate your data on read/write.

```python
class Ticket(BaseModel):
    id: str
    title: str
    status: str
    description: str
```

### 3\. Upsert (Write)

This saves the JSON **AND** automatically generates a vector embedding for the `description` and `title`.

```python
tickets = db.collection("tickets", Ticket)

# Upsert (Insert or Update)
new_ticket = Ticket(
    id="t-100", 
    title="Login Broken", 
    status="open", 
    description="User cannot reset password on mobile."
)
tickets.upsert(new_ticket)
```

### 4\. Semantic Search (RAG)

Find records by *meaning*, not just keywords.

```python
# Finds the ticket above because "password reset" is related to "login"
results = tickets.search("Issues with authentication", limit=1)

print(results[0].title) 
# Output: "Login Broken"
```

### 5\. Management

```python
# Get by ID
ticket = tickets.read("t-100")

# Delete
tickets.delete("t-100")

# List all collections
print(db.list_collections())
# Output: ['tickets', 'users']
```

-----

## AI Agent Integration (Claude / MCP)

FlowDB is **MCP-Native**. This allows **Claude Desktop** to natively "see", "search", and "edit" your database without any custom glue code.

**What this enables:**
You can ask Claude: *"Check the 'tickets' collection for any high-priority bugs regarding login issues, and summarize them for me."*

### Step 1: Install the Bridge Tool

We use `fastmcp` to bridge Claude (Stdio) to FlowDB (HTTP).

```bash
pip install fastmcp
```

### Step 2: Configure Claude

Edit your config file:

  * **Mac:** `~/Library/Application Support/Claude/claude_desktop_config.json`
  * **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

Update it like this:

```json
{
  "preferences": {
    "menuBarEnabled": false,
    "quickEntryShortcut": "off"
  },
  "mcpServers": {
    "flowdb": {
      "command": "/Users/<user>/.pyenv/shims/fastmcp", // path-to-fastmcp
      "args": [
        "run",
        "http://:<API_KEY>@localhost:8000/mcp/sse"
      ]
    }
  }
}


```

> **Note on Security:** We pass the API Key in the URL format `http://:PASSWORD@HOST` (Basic Auth style). Replace `YOUR_API_KEY` with the key from your `.env` file. If you haven't set a key (Dev Mode), just use `http://localhost:8000/mcp/sse`.

-----

## N8n & Automation Integration

FlowDB is perfect for low-code automation because it accepts raw JSON. It acts as the "Long Term Memory" for your workflows.

**In N8n:**

1.  Use the **HTTP Request** node.
2.  **Method:** `POST`
3.  **URL:** `http://your-server:8000/v1/users/upsert`
4.  **Auth:** Header Auth (`Authorization: Bearer YOUR_KEY`).
5.  **Body:** Pass your raw JSON from Typeform/Gmail directly into the `data` field.

-----

## Architecture

  * **Storage Engine:** [LMDB](https://www.google.com/search?q=https://www.lmdb.tech/) (Lightning Memory-Mapped Database). A transactional key-value store that runs at the speed of RAM.
  * **Vector Engine:** [USearch](https://github.com/unum-cloud/usearch). A high-performance HNSW implementation for semantic search.
  * **API Layer:** [FastAPI](https://fastapi.tiangolo.com/). Handles concurrent reads and serialized writes.
  * **Protocol:** [MCP](https://modelcontextprotocol.io/). Native agentic interface.

-----

## License

Apache 2.0 License. Built for the AI Community.