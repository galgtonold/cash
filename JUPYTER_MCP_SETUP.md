# Jupyter MCP Server Configuration Guide

## Overview
This guide helps you configure the Jupyter MCP Server to work with MCP clients like Claude Desktop, Cursor, Windsurf, etc.

## Prerequisites
✅ All packages installed successfully:
- jupyterlab==4.4.1
- jupyter-collaboration==4.0.2
- jupyter-mcp-tools>=0.1.4
- ipykernel
- datalayer_pycrdt==0.12.17
- jupyter-mcp-server==0.22.1

## JupyterLab Server
The JupyterLab server is running at:
- **URL**: http://localhost:8888
- **Token**: MY_TOKEN

## Configuration Options

### Option 1: Using uvx (Recommended for Quick Start)

Add this to your MCP client configuration (e.g., Claude Desktop's `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "jupyter": {
      "command": "uvx",
      "args": [
        "jupyter-mcp-server",
        "--jupyter-url", "http://localhost:8888",
        "--jupyter-token", "MY_TOKEN",
        "--jupyterlab", "true",
        "--document-id", "test_mcp_notebook.ipynb"
      ]
    }
  }
}
```

### Option 2: Using Python Virtual Environment

If you prefer to use the local Python environment:

```json
{
  "mcpServers": {
    "jupyter": {
      "command": "C:/Users/Philipp/My Drive/Cloud/PycharmProjects/cash/.venv/Scripts/python.exe",
      "args": [
        "-m",
        "jupyter_mcp_server",
        "--jupyter-url", "http://localhost:8888",
        "--jupyter-token", "MY_TOKEN",
        "--jupyterlab", "true",
        "--document-id", "test_mcp_notebook.ipynb"
      ]
    }
  }
}
```

### Option 3: Using Docker (For Production)

1. Build the Docker image:
```powershell
docker build -t jupyter-mcp-server .
```

2. Run the container:
```powershell
docker run -p 8765:8765 jupyter-mcp-server --jupyter-url http://host.docker.internal:8888 --jupyter-token MY_TOKEN --transport streamable-http --port 8765
```

3. Configure your MCP client to connect via HTTP:
```json
{
  "mcpServers": {
    "jupyter": {
      "url": "http://localhost:8765"
    }
  }
}
```

## Configuration Parameters Explained

- `--jupyter-url`: The URL of your JupyterLab server (http://localhost:8888)
- `--jupyter-token`: Authentication token for the Jupyter server (MY_TOKEN)
- `--jupyterlab`: Enable JupyterLab mode (true/false) - enables UI integration
- `--document-id`: Path to the notebook to connect to (optional - can be selected interactively)
- `--transport`: Communication method (stdio or streamable-http)
- `--allowed-jupyter-mcp-tools`: Comma-separated list of enabled tools (default: notebook_run-all-cells,notebook_get-selected-cell)

## Testing the Connection

### 1. Verify JupyterLab is running
Open http://localhost:8888/?token=MY_TOKEN in your browser

### 2. Test the MCP Server manually
Run this command to test the server:
```powershell
& "C:/Users/Philipp/My Drive/Cloud/PycharmProjects/cash/.venv/Scripts/python.exe" -m jupyter_mcp_server --jupyter-url http://localhost:8888 --jupyter-token MY_TOKEN --document-id test_mcp_notebook.ipynb
```

### 3. Available MCP Tools
Once connected, your MCP client will have access to these tools:

**Server Management:**
- `list_files` - List files and directories in the Jupyter server
- `list_kernels` - List all available and running kernel sessions
- `connect_to_jupyter` - Connect to a Jupyter server dynamically

**Multi-Notebook Management:**
- `use_notebook` - Connect to a notebook file or create a new one
- `list_notebooks` - List all notebooks available on the server
- `restart_notebook` - Restart the kernel for a specific notebook
- `unuse_notebook` - Disconnect from a specific notebook
- `read_notebook` - Read notebook cells source content

**Cell Operations:**
- `read_cell` - Read the full content of a single cell
- `insert_cell` - Insert a new code or markdown cell
- `delete_cell` - Delete a cell at a specified index
- `overwrite_cell_source` - Overwrite the source code of an existing cell
- `execute_cell` - Execute a cell with timeout support
- `insert_execute_code_cell` - Insert and execute a code cell in one step
- `execute_code` - Execute code directly in the kernel

**JupyterLab Integration (when enabled):**
- `notebook_run-all-cells` - Execute all cells in the notebook
- `notebook_get-selected-cell` - Get information about the currently selected cell

## Example Prompts to Test

Once connected to your MCP client:

1. "List all notebooks in the Jupyter server"
2. "Read the contents of test_mcp_notebook.ipynb"
3. "Execute cell 1 in the notebook"
4. "Insert a new cell that calculates the mean of df['A']"
5. "Run all cells in the notebook"

## Troubleshooting

### Real-time Collaboration Check
To verify your environment is configured correctly:
1. Open a notebook in JupyterLab
2. Type content in any cell
3. You should see an "×" appear next to the notebook name
4. Wait a few seconds - it should automatically change to "●" without saving
5. This confirms real-time collaboration features are working

### Common Issues

**Port conflicts:**
If port 8888 is in use, change it in both the jupyter lab command and MCP config:
```powershell
jupyter lab --port 9999 --IdentityProvider.token MY_TOKEN --ip 0.0.0.0 --no-browser
```

**Token authentication:**
Make sure the token in your MCP config matches the one used to start JupyterLab.

**Notebook path:**
The `document-id` should be relative to the directory where JupyterLab was started (the cash project root).

## Next Steps

1. Configure your preferred MCP client using one of the options above
2. Open the test notebook at http://localhost:8888/?token=MY_TOKEN
3. Start interacting with your notebook through your MCP client!

## Additional Resources

- [Jupyter MCP Server Documentation](https://jupyter-mcp-server.datalayer.tech)
- [GitHub Repository](https://github.com/datalayer/jupyter-mcp-server)
- [MCP Protocol Specification](https://modelcontextprotocol.io)
