## Client SDK

FlowDB comes with a Python Client SDK. It feels like using a local dictionary, but it talks to your high-performance
server.

### 1\. Initialize the db

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
# tickets is a FlowDB collection of Ticket objects
# It acts like a local dictionary and allows you to use props on the object
tickets = db.collection("tickets", Ticket)

# Upsert (Insert or Update)
new_ticket = Ticket(
    id="t-100",
    title="Login Broken",
    status="open",
    description="User cannot reset password on mobile."
)

# Upsert writes the data, either replacing existing content with the same id or creating a new object
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
# Returns a Ticket object with matching ID or None

# Delete
tickets.delete("t-100")

# List objects in tickets collection with pagination
tickets.list(limit=7, skip=0)

# List all collections
print(db.list_collections())
# Output: ['tickets', 'users']
```