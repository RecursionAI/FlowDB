## Using the MCP server

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