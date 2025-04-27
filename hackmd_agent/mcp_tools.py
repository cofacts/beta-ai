# hackmd_agent/tools.py
import asyncio
import os
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from contextlib import AsyncExitStack
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define MCP server configurations
MCP_SERVER_CONFIGS = {
    "hackmd": {
        "command": "npx",
        "args": ["-y", "hackmd-mcp"],
        "env": {
            "HACKMD_API_URL": os.getenv("HACKMD_API_URL"),
            "HACKMD_API_TOKEN": os.getenv("HACKMD_API_TOKEN"),
        }
    },
    "github": {
        "command": "docker",
        "args": [
            "run", "-i", "--rm",
            "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
            "ghcr.io/github/github-mcp-server"
        ],
        "env": {
            "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"),
        }
    }
}

async def get_mcp_tools_async():
    """
    Fetches tools from configured MCP servers (e.g., HackMD, GitHub).

    Relies on ADK's or the environment's automatic .env loading.

    Returns:
        tuple: A tuple containing the combined list of tools and an AsyncExitStack
               for managing server connections. (list[BaseTool], AsyncExitStack)
    """
    combined_tools = []
    # Use AsyncExitStack to manage multiple asynchronous contexts (MCP server connections)
    exit_stack = AsyncExitStack()

    for server_name, config in MCP_SERVER_CONFIGS.items():
        logging.info(f"Processing MCP server configuration for: {server_name}")

        # Check if required environment variables are set
        env_dict = config.get("env", {})
        if not all(env_dict.values()):
            missing_vars = [var for var, value in env_dict.items() if not value]
            logging.warning(f"Skipping {server_name} MCP server: Missing environment variables: {missing_vars}")
            continue # Skip this server if required env vars are missing

        logging.info(f"Attempting to connect to {server_name} MCP server...")
        try:
            # Construct connection parameters using the configuration
            connection_params = StdioServerParameters(
                command=config["command"],
                args=config["args"],
                env=env_dict
            )

            # Call from_server and get tools and the server's own exit stack
            tools, server_stack = await MCPToolset.from_server(
                connection_params=connection_params
            )
            logging.info(f"Successfully fetched {len(tools)} tools from {server_name}.")
            combined_tools.extend(tools)
            # Add the server's exit stack to the main exit_stack for management
            await exit_stack.enter_async_context(server_stack)
            logging.info(f"Added {server_name} server connection to exit stack.")

        except Exception as e:
            # Log the error along with traceback for better debugging
            logging.error(f"Error connecting to or fetching tools from {server_name} MCP server: {e}", exc_info=True)

    logging.info(f"Total MCP tools fetched: {len(combined_tools)}")
    # Return the combined list of tools and the exit_stack managing connections
    # Note: The caller of this function is responsible for calling exit_stack.aclose() later.
    return combined_tools, exit_stack

# Example usage block for testing the script directly (optional)
async def _test_run():
    """Internal function to test get_mcp_tools_async directly."""
    print("--- Testing get_mcp_tools_async ---")
    # Ensure necessary environment variables are set in your testing environment
    tools, stack = await get_mcp_tools_async()
    print(f"--- Test Result: Got {len(tools)} tools ---")
    tool_names = [tool.name for tool in tools]
    print(f"Tool names: {tool_names}")
    print("--- Closing connections... ---")
    await stack.aclose()
    print("--- Connections closed. Test finished. ---")

# This allows testing the script directly using `uv run --env-file .env tools.py`
#
if __name__ == "__main__":
    asyncio.run(_test_run())
