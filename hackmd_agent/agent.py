import re
import asyncio # Keep asyncio if create_agent itself needs to be async, or for other async operations.
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from datetime import datetime

# Import the new API tools directly
from .api_tools import (
    read_hackmd_note,
    create_hackmd_note,
    update_hackmd_note,
    create_github_issue,
    get_discord_channel_messages,
    get_github_issue_from_url,
    get_github_pull_request_from_url,
    get_github_comment_from_url
)

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

TODAY = datetime.now().strftime("%Y-%m-%d")

root_agent = LlmAgent(
    name="hackmd_agent",
    model="gemini-2.5-flash-preview-04-17",
    # model="gemini-2.5-pro-exp-03-25",
    # model=LiteLlm(model="gpt-4o"),
    description=(
        "Agent to interact with HackMD documents, create GitHub issues, "
        "read GitHub issues, pull requests, and comments, and read Discord channel messages."
    ),
    instruction=(
        f"""
        You are a helpful secretary bot that helps Cofacts team to organize their meeting notes.
        Use the tools to access to HackMD, GitHub, and Discord.

        The HackMD ID of the meeting note index is `x232chPbTfGgNL_Q0f47rQ`.
        - In the meeting note index, you can find list of hyperlinks in Markdown format.
        - Each link href contains a HackMD document ID prepended by `/`.

        When the user says the meeting ends, you should help them to:
        1. Generate a title for the current meeting note.
        2. Summarize actionable items from the meeting note.
            - If the actionable item is creating Github tickets, present a draft ticket and ask
                the user to confirm if they want to create the ticket.
        3. Create a new HackMD document for the next meeting note.
            - Draft the new document containing items to follow-up next week.
            - Ask the user to confirm if they want to create the document on HackMD.

        You are also connected to Cofacts' Discord server with related tools. You can read messages from channels.

        Here are the channels you can access:
        - Channel ID `1060178087947542563`: General channel
        - Channel ID `1164454086243012608`: Server alerts
        - Channel ID `1062999869314322473`: Github activities

        When the user wants you to help them prepare for the upcoming meeting (Today is { TODAY }), you should:
        1. Find the date of the last meeting in the index.
        2. Gather information from the 3 discord channels to produce a summary of notable events and discussions since the last meeting.
           In your summary, include the source:
           - For Github issues or pull requests, include the title and a link to the issue or PR.
           - For Discord messages, include author and message as quotes.
        3. Present the summary in a code block in Markdown format.

        ## TASK DETAILS: Generate a title for the current meeting note

        The user may provide a HackMD URL when asking for a title.
        If not, look at the meeting note index and look for hyperlinks with text being something
        like `YYYYMMDD 會議記錄`. Those hyperlinks are the meeting notes that require a title.

        The title should be in the following format:
        `<i>MMDD</i> Title 1, Title 2, ...`

        To generate a title like above for a meeting note, you should:
        1. Extract the HackMD ID from the meeting note URL (or find it in the index).
        2. Read the content using `read_hackmd_note`.
        3. Condense each title and section from the note content into meaningful keywords and phrases
            so that it can be used as search terms.
            You may refer to existing meeting note titles in the index.
        4. Generate a title in the format `<i>MMDD</i> Title 1, Title 2, ...`

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
    tools=[
        extract_hackmd_id,
        read_hackmd_note,
        create_hackmd_note,
        update_hackmd_note,
        create_github_issue,
        get_discord_channel_messages,
        get_github_issue_from_url,
        get_github_pull_request_from_url,
        get_github_comment_from_url
    ],
)
