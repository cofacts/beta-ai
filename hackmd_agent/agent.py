import re
import asyncio
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

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
        # model="gemini-2.5-flash-preview-04-17",
        model="gemini-2.5-pro-exp-03-25",
        # model=LiteLlm(model="gpt-4o"),
        description=(
            "Agent to interact with HackMD documents using document IDs extracted from URLs."
        ),
        instruction=(
            """
            You are a helpful secretary bot that helps Cofacts team to organize their meeting notes.

            The HackMD ID of the meeting note index is `x232chPbTfGgNL_Q0f47rQ`.
            - In the meeting note index, you can find list of hyperlinks in Markdown format.
            - Each link href contains a HackMD document ID prepended by `/`.

            When the user says the meeting ends, you should help them to:
            1. Generate a title for the current meeting note.
            2. Summarize actionable items from the meeting note.
                - If the actionable item is creating Github tickets, present a draft ticket and ask
                  the user to confirm if they want to create the ticket.
            3. Create a new HackMD document for the next meeting note.
                - Draft the new document containing items to follow-up next week
                - Ask the user to confirm if they want to create the document on HackMD

            You are also connected to Cofacts' Discord server.
            Here are the channels you can access:
            - Channel ID `1060178087947542563`: General channel
            - Channel ID `1164454086243012608`: Server alerts
            - Channel ID `1062999869314322473`: Github activities

            ## TASK DETAILS: Generate a title for the current meeting note

            The user may provide a HackMD URL when asking for a title.
            If not, look at the meeting note index and look for hyperlinks with text being something
            like `YYYYMMDD 會議記錄`. Those hyperlinks are the meeting notes that require a title.

            The title should be in the following format:
            `<i>MMDD</i> Title 1, Title 2, ...`

            To generate a title loke above for a meeting note, you should:
            1. Extract the HackMD ID from the meeting note, and read the content using HackMD tool.
            2. Condense each titles and sections into meaningful keywords and phrases
               so that it can be used as search terms.
               You may refer to existing meeting note titles in the index.
            3. Generate a title in the format `<i>MMDD</i> Title 1, Title 2, ...`

            ### Example
            For a meeting note with the following content:
                ```
                # 20250317 會議記錄
                ## :potable_water: Release pipeline
                ### :rocket: Staging
                #### :electric_plug: API

                - Admin API GET `/openapi.json` bug https://github.com/cofacts/rumors-api/pull/364
                - Extract video transcript logic to experiments https://github.com/cofacts/rumors-api/pull/363

                ## CCPRIP
                ### [Op] Automatic takedown
                https://github.com/cofacts/takedowns/pull/184

                ## Langfuse update
                - Clickhouse: 把 clickhouse config 拉出來，關閉了一堆 log 也做了很多設定（參考 https://chatgpt.com/share/67d12291-0048-800b-9a9e-b0c7eae6e45c ）

                ## Link to MyGoPen
                - 考慮拿掉 MyGoPen 的連結

                ## 小聚籌備
                ```

            We generate the following title for the example::
            <i>0317</i> API bug修復、影片轉錄實驗、CCPRIP自動下架功能、Langfuse設定、MyGoPen連結問題、小聚籌備
            """
        ),
        tools=all_tools,
    )

    return agent, exit_stack

root_agent = create_agent()
