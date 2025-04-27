import re
import asyncio
from google.adk.agents import LlmAgent

# Import our MCP tools fetcher
from .mcp_tools import get_mcp_tools_async

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
    pattern = r'(?:\/|%2F)?([-\w]{16,22})(?:\/|$)'

    # Check if input is null or empty
    if not url:
        return {
            "status": "error",
            "error_message": "No URL provided. Please provide a HackMD URL."
        }

    # Search for the pattern in the URL
    match = re.search(pattern, url)

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


# From: https://google.github.io/adk-docs/tools/mcp-tools/#mcp-with-adk-web
async def create_agent():
    """Get an agent that includes both local and MCP tools."""

    mcp_tools, exit_stack = await get_mcp_tools_async()

    all_tools = [extract_hackmd_id] + mcp_tools

    # Create and return agent with all tools
    agent = LlmAgent(
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
        tools=all_tools,
    )

    return agent, exit_stack

root_agent = create_agent()
