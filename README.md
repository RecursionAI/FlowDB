# FlowDB

FlowDB is a JSON-based database stack that is AI-Native.

It's core objective is to be fast, simple, and n8n + AI optimized.

-----

## Core Features

- JSON (No-SQL)
- API controlled for n8n-usage and database safety
- API Key protected
- Dockerized for ease of use
- MCP server for Agentic use
- Vectorization built in (BYOK)
- Client CLI and PyPi package
- Speed: Built on LMDB (Lightning Memory-Mapped Database) and USearch (HNSW Vector Search).

-----

## Guidelines

FlowDB is intentionally built to be fast, simple, and AI-Native. It is **NOT** ideal for compliance applications such as
FinTech because it lacks SQLs ACID safety measures.

FlowDB is ideal for AI applications, automation, speed, and n8n workflows.

-----

## Getting Started

### [Install and Setup](./docs/install.md)

Full guide to install both the Client SDK and configure the dockerized server.

### [Client SDK](./docs/sdk.md)

How to use the client SDK to interact with the database in your code and utilyze Pydantic models

### [Full n8n Guide](./docs/n8n.md)

Comprehensive guide for implementing FlowDB in your n8n workflows for and open-source persistent memory adn VectorDB
solution.

### [URL Schemas](./docs/schemas.md)

URL schemas and API guidelines.


### [MCP Server](./docs/mcp.md)
How the MCP server works and how to set it up so Claude Desktop can use it and interact with your db.

-----
