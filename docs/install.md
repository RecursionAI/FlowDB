
### Installation

#### Client SDK:

```bash
pip install flowdb
```

#### Server:
git clone the repo to spin up your local dev server or production dockerized server.

-----

### Configuration

FlowDB is configured via a `.env` file in your project root or via Environment Variables.

This is where you set your OpenAI API key for auto-vectorization and your FlowDB API key for database protection.

| Variable            | Description                                            | Required? | Default              |
|:--------------------|:-------------------------------------------------------|:----------|:---------------------|
| `OPENAI_API_KEY`    | Your OpenAI Key for auto-vectorization.                | **Yes**   | `None`               |
| `FLOWDB_API_KEY`    | Secures the DB. If set, clients must provide this key. | No        | `None` (Open Access) |
| `FLOWDB_PATH`       | Where data files are stored on disk.                   | No        | `./flow_data`        |
| `FLOWDB_VECTORIZER` | Provider to use (`openai` or `mock`).                  | No        | `auto`               |

**Example `.env` file:**

```ini
OPENAI_API_KEY = sk-proj-12345...
FLOWDB_API_KEY = my-super-secret-password
FLOWDB_PATH = ./my_db_storage
FLOWDB_VECTORIZER = openai
```

-----

For running the server locally simply run:

```bash
flowdb start
```

For production navigate to the root FlowDB directory and run:

```bash
docker-compose up -d
```

This creates an API on port 8000 ready for reads and writes. The client SDK will automaticlaly read and write to this
port. If using API key configure a FLOWDB_API_KEY in your `.env` file and the client SDK should just work.