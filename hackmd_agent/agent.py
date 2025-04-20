import asyncio
import re
import logging
from google.adk.agents import Agent

# Import our custom MCP tools fetcher
from .tools import get_mcp_tools_async

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_hackmd_id(url: str) -> dict:
    """
    Extracts HackMD document ID from various URL formats.

    Supports formats like:
    - https://g0v.hackmd.io/@cofacts/meetings/%2F-GmgAfesTB6n1pxGvWQvWA
    - https://g0v.hackmd.io/SRVhEtOTQf-mQSV7CVstkw
    - Links containing /SRVhEtOTQf-mQSV7CVstkw format

    Args:
        url (str): The HackMD URL or link.

    Returns:
        dict: A dictionary containing status and either the HackMD ID or an error message.
    """
    # Try to match different HackMD URL patterns
    # Pattern 1: URLs ending with a HackMD ID (with or without %2F prefix)
    pattern1 = r'(?:\/|%2F)?([-\w]{16,22})(?:\/|$)'

    # Check if input is null or empty
    if not url:
        return {
            "status": "error",
            "error_message": "No URL provided. Please provide a HackMD URL."
        }

    # Search for the pattern in the URL
    match = re.search(pattern1, url)

    if match:
        hackmd_id = match.group(1)
        return {
            "status": "success",
            "hackmd_id": hackmd_id,
            "message": f"Successfully extracted HackMD ID: {hackmd_id}"
        }
    else:
        return {
            "status": "error",
            "error_message": f"Could not extract a HackMD ID from the provided URL: {url}"
        }

# Helper function to asynchronously get the agent with all tools
async def _get_agent_with_mcp_tools():
    """
    Internal helper function to create the agent with MCP tools.
    Used to initialize root_agent.

    Returns:
        tuple: (Agent, AsyncExitStack)
    """
    try:
        # Get MCP tools (primarily HackMD tools)
        mcp_tools, exit_stack = await get_mcp_tools_async()
        logging.info(f"Loaded {len(mcp_tools)} MCP tools")

        # Create the agent with our URL extractor tool and MCP tools
        agent = Agent(
            name="hackmd_agent",
            model="gemini-2.0-flash",
            description=(
                "Agent to interact with HackMD documents using document IDs extracted from URLs."
            ),
            instruction=(
                "You are a helpful agent who can extract HackMD document IDs from URLs "
                "and interact with HackMD documents using the available tools. "
                "When given a HackMD URL, first extract the document ID using the extract_hackmd_id tool, "
                "then use the appropriate HackMD tools to perform operations on that document."
            ),
            tools=[extract_hackmd_id] + mcp_tools,
        )

        return agent, exit_stack
    except Exception as e:
        logging.error(f"Error initializing agent with MCP tools: {e}", exc_info=True)
        # Fallback to basic agent without MCP tools
        return Agent(
            name="hackmd_agent_fallback",
            model="gemini-2.0-flash",
            description="Agent to extract HackMD document IDs from URLs.",
            instruction="You are a helpful agent who can extract HackMD document IDs from URLs.",
            tools=[extract_hackmd_id],
        ), None

# Initialize the root_agent asynchronously during module import
try:
    # Create an event loop to run the async initialization
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    root_agent, _exit_stack = loop.run_until_complete(_get_agent_with_mcp_tools())
    loop.close()

    # Note: We're intentionally not closing the exit_stack, as it needs to stay open
    # while the agent is in use. In a production app, you'd need a way to properly
    # clean up when the app is shutting down.
    logging.info("Successfully initialized root_agent with MCP tools")
except Exception as e:
    logging.error(f"Failed to initialize agent with MCP tools: {e}", exc_info=True)
    # Fallback to basic agent without MCP tools if async initialization fails
    root_agent = Agent(
        name="hackmd_agent_basic",
        model="gemini-2.0-flash",
        description="Agent to extract HackMD document IDs from URLs.",
        instruction="You are a helpful agent who can extract HackMD document IDs from URLs.",
        tools=[extract_hackmd_id],
    )
    logging.info("Initialized root_agent with basic functionality (no MCP tools)")

