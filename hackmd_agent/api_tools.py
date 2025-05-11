# hackmd_agent/api_tools.py
import httpx
import os
import json
from typing import Optional, Dict, Any, List

# Attempt to get base URL and token from environment variables
HACKMD_API_URL = os.getenv("HACKMD_API_URL", "https://api.hackmd.io/v1") # Default if not set
HACKMD_API_TOKEN = os.getenv("HACKMD_API_TOKEN")

async def _make_hackmd_request(method: str, endpoint: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Helper function to make requests to the HackMD API."""
    if not HACKMD_API_TOKEN:
        return {"status": "error", "error_message": "HACKMD_API_TOKEN is not set in environment variables."}

    headers = {
        "Authorization": f"Bearer {HACKMD_API_TOKEN}",
        "Content-Type": "application/json"
    }
    url = f"{HACKMD_API_URL.rstrip('/')}/{endpoint.lstrip('/')}"

    async with httpx.AsyncClient() as client:
        try:
            response = None
            if method.upper() == "GET":
                response = await client.get(url, headers=headers)
            elif method.upper() == "POST":
                response = await client.post(url, headers=headers, json=payload)
            elif method.upper() == "PATCH":
                response = await client.patch(url, headers=headers, json=payload)
            else:
                return {"status": "error", "error_message": f"Unsupported HTTP method: {method}"}

            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            return {"status": "success", "data": response.json()}
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text
            try:
                error_json = e.response.json()
                error_detail = error_json.get("message", error_detail)
            except json.JSONDecodeError:
                pass # Use text if JSON decoding fails
            return {"status": "error", "error_message": f"API request failed: {e.response.status_code} - {error_detail}", "status_code": e.response.status_code}
        except httpx.RequestError as e:
            return {"status": "error", "error_message": f"Request error: {str(e)}"}
        except Exception as e:
            return {"status": "error", "error_message": f"An unexpected error occurred: {str(e)}"}

async def read_hackmd_note(note_id: str) -> Dict[str, Any]:
    """
    Reads the content and metadata of a specific HackMD note given its ID.
    Corresponds to 'get_note' from the original HackMD MCP server.

    Args:
        note_id (str): The HackMD note ID to read.

    Returns:
        dict: A dictionary containing the status and the note data if successful, or an error message.
              Example success: {"status": "success", "note_data": {"id": "...", "title": "...", "content": "...", ...}}
              Example error: {"status": "error", "error_message": "Failed to read note."}
    """
    result = await _make_hackmd_request("GET", f"notes/{note_id}")
    if result["status"] == "success":
        return {"status": "success", "note_data": result["data"]}
    return result

async def create_hackmd_note(
    title: Optional[str] = None,
    content: Optional[str] = None,
    read_permission: Optional[str] = None, # e.g., "owner", "signed_in", "guest"
    write_permission: Optional[str] = None, # e.g., "owner", "signed_in", "guest"
    comment_permission: Optional[str] = None, # e.g., "disabled", "forbidden", "owners", "signed_in_users", "everyone"
    permalink: Optional[str] = None
) -> Dict[str, Any]:
    """
    Creates a new HackMD note. All parameters are optional.
    Permission values should match HackMD API specifications.
    Corresponds to 'create_note' from the original HackMD MCP server.

    Args:
        title (Optional[str]): The title of the note.
        content (Optional[str]): The content of the note.
        read_permission (Optional[str]): Read permission for the note.
        write_permission (Optional[str]): Write permission for the note.
        comment_permission (Optional[str]): Comment permission for the note.
        permalink (Optional[str]): Custom permalink for the note.

    Returns:
        dict: A dictionary containing the status and the new note data if successful, or an error message.
              Example success: {"status": "success", "note_data": {"id": "new_note_id", "title": "...", ...}}
    """
    payload = {}
    if title is not None:
        payload["title"] = title
    if content is not None:
        payload["content"] = content
    if read_permission is not None:
        payload["readPermission"] = read_permission
    if write_permission is not None:
        payload["writePermission"] = write_permission
    if comment_permission is not None:
        payload["commentPermission"] = comment_permission # API might use different casing or structure
    if permalink is not None:
        payload["permalink"] = permalink

    if not payload: # If no actual content to create, maybe return an error or specific message
        return {"status": "error", "error_message": "Cannot create an empty note. Please provide title, content, or other attributes."}

    result = await _make_hackmd_request("POST", "notes", payload=payload)
    if result["status"] == "success":
        return {"status": "success", "note_data": result["data"]}
    return result

async def update_hackmd_note(
    note_id: str,
    content: Optional[str] = None,
    read_permission: Optional[str] = None,
    write_permission: Optional[str] = None,
    permalink: Optional[str] = None
    # title update might also be possible via PATCH, depending on API
) -> Dict[str, Any]:
    """
    Updates an existing HackMD note. 'note_id' is required.
    Other parameters are optional and specify the attributes to update.
    Permission values should match HackMD API specifications.
    Corresponds to 'update_note' from the original HackMD MCP server.

    Args:
        note_id (str): The ID of the HackMD note to update.
        content (Optional[str]): The new content for the note.
        read_permission (Optional[str]): The new read permission.
        write_permission (Optional[str]): The new write permission.
        permalink (Optional[str]): The new custom permalink.

    Returns:
        dict: A dictionary containing the status of the operation.
              Example success: {"status": "success", "message": "Note updated successfully."}
              Example (if API returns updated note): {"status": "success", "note_data": {"id": "...", ...}}
    """
    payload = {}
    if content is not None:
        payload["content"] = content
    if read_permission is not None:
        payload["readPermission"] = read_permission
    if write_permission is not None:
        payload["writePermission"] = write_permission
    if permalink is not None:
        payload["permalink"] = permalink
    # Add other updatable fields like title if supported by PATCH /notes/{id}

    if not payload:
        return {"status": "info", "message": "No update parameters provided for the note."}

    result = await _make_hackmd_request("PATCH", f"notes/{note_id}", payload=payload)
    if result["status"] == "success":
        # Some APIs return the updated resource, others just a success status
        if result.get("data"):
             return {"status": "success", "note_data": result["data"]}
        return {"status": "success", "message": f"Note {note_id} updated successfully."}
    return result

# GitHub API Configuration
GITHUB_API_URL = "https://api.github.com"
GITHUB_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN") # Ensure this matches your .env
GITHUB_API_VERSION = "2022-11-28"

async def _make_github_request(method: str, endpoint: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Helper function to make requests to the GitHub API."""
    if not GITHUB_TOKEN:
        return {"status": "error", "error_message": "GITHUB_PERSONAL_ACCESS_TOKEN is not set in environment variables."}

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
        "Content-Type": "application/json"
    }
    url = f"{GITHUB_API_URL.rstrip('/')}/{endpoint.lstrip('/')}"

    async with httpx.AsyncClient() as client:
        try:
            response = None
            if method.upper() == "POST":
                response = await client.post(url, headers=headers, json=payload)
            # Add other methods like GET, PATCH if needed for future GitHub tools
            else:
                return {"status": "error", "error_message": f"Unsupported HTTP method for GitHub: {method}"}

            response.raise_for_status()
            return {"status": "success", "data": response.json()}
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text
            try:
                error_json = e.response.json() # GitHub errors are usually JSON
                error_detail = error_json.get("message", error_detail)
                if "errors" in error_json: # More detailed errors
                    error_detail += f" Details: {json.dumps(error_json['errors'])}"
            except json.JSONDecodeError:
                pass
            return {"status": "error", "error_message": f"GitHub API request failed: {e.response.status_code} - {error_detail}", "status_code": e.response.status_code}
        except httpx.RequestError as e:
            return {"status": "error", "error_message": f"GitHub request error: {str(e)}"}
        except Exception as e:
            return {"status": "error", "error_message": f"An unexpected error occurred with GitHub request: {str(e)}"}

async def create_github_issue(
    owner: str,
    repo: str,
    title: str,
    body: Optional[str] = None,
    assignees: Optional[List[str]] = None,
    labels: Optional[List[str]] = None
    # milestone: Optional[int] = None, # Can be added if needed
) -> Dict[str, Any]:
    """
    Creates a new issue in the specified GitHub repository.

    Args:
        owner (str): The owner of the repository (e.g., "octocat").
        repo (str): The name of the repository (e.g., "Spoon-Knife").
        title (str): The title of the issue.
        body (Optional[str]): The contents of the issue.
        assignees (Optional[List[str]]): Logins for users to assign to this issue.
        labels (Optional[List[str]]): Labels to associate with this issue.

    Returns:
        dict: A dictionary containing the status and the new issue data (e.g., URL, number) if successful,
              or an error message.
              Example success: {"status": "success", "issue_data": {"html_url": "...", "number": 123, ...}}
    """
    if not owner or not repo or not title:
        return {"status": "error", "error_message": "Owner, repo, and title are required to create a GitHub issue."}

    payload = {"title": title}
    if body is not None:
        payload["body"] = body
    if assignees:
        payload["assignees"] = assignees
    if labels:
        payload["labels"] = labels
    # if milestone is not None:
    #     payload["milestone"] = milestone

    endpoint = f"repos/{owner}/{repo}/issues"
    result = await _make_github_request("POST", endpoint, payload=payload)

    if result["status"] == "success":
        return {"status": "success", "issue_data": result["data"]}
    return result

# Discord API Configuration
DISCORD_API_URL = "https://discord.com/api/v10"  # Using v10, adjust if needed
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN") # Ensure this matches your .env

async def _make_discord_request(method: str, endpoint: str, params: Optional[Dict[str, Any]] = None, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Helper function to make requests to the Discord API."""
    if not DISCORD_BOT_TOKEN:
        return {"status": "error", "error_message": "DISCORD_BOT_TOKEN is not set in environment variables."}

    headers = {
        "Authorization": f"Bot {DISCORD_BOT_TOKEN}", # Note "Bot" prefix
        "Content-Type": "application/json"
    }
    url = f"{DISCORD_API_URL.rstrip('/')}/{endpoint.lstrip('/')}"

    async with httpx.AsyncClient() as client:
        try:
            response = None
            if method.upper() == "GET":
                response = await client.get(url, headers=headers, params=params)
            elif method.upper() == "POST": # Added for completeness, though not used by get_discord_channel_messages
                response = await client.post(url, headers=headers, json=payload, params=params)
            # Add other methods like PATCH, DELETE if needed for future Discord tools
            else:
                return {"status": "error", "error_message": f"Unsupported HTTP method for Discord: {method}"}

            response.raise_for_status()
            # For 204 No Content, response.json() will fail.
            if response.status_code == 204:
                return {"status": "success", "data": None}
            return {"status": "success", "data": response.json()}
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text
            try:
                error_json = e.response.json()
                error_detail = error_json.get("message", error_detail)
                if "errors" in error_json:
                     error_detail += f" Details: {json.dumps(error_json['errors'])}"
            except json.JSONDecodeError:
                pass
            return {"status": "error", "error_message": f"Discord API request failed: {e.response.status_code} - {error_detail}", "status_code": e.response.status_code}
        except httpx.RequestError as e:
            return {"status": "error", "error_message": f"Discord request error: {str(e)}"}
        except Exception as e:
            return {"status": "error", "error_message": f"An unexpected error occurred with Discord request: {str(e)}"}

async def get_discord_channel_messages(
    channel_id: str,
    limit: int = 50,
    # before: Optional[str] = None,
    # after: Optional[str] = None,
    # around: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fetches recent messages from a specified Discord channel.
    Your agent's prompt should guide the LLM on how to summarize these messages if needed.

    Args:
        channel_id (str): The ID of the Discord channel to fetch messages from.
        limit (int): The maximum number of messages to return (1-100). Defaults to 50.
        # before (Optional[str]): Get messages before this message ID. (For pagination)
        # after (Optional[str]): Get messages after this message ID. (For pagination)
        # around (Optional[str]): Get messages around this message ID. (For pagination)

    Returns:
        dict: A dictionary containing the status and a list of messages if successful, or an error message.
              Each message in the list is a dictionary with details like id, author, content, timestamp.
              Example success: {"status": "success", "messages": [{"id": "...", "author": {...}, "content": "...", ...}]}
    """
    if not channel_id:
        return {"status": "error", "error_message": "Channel ID is required to fetch Discord messages."}

    actual_limit = max(1, min(limit, 100)) # Clamp limit between 1 and 100

    params = {"limit": actual_limit}
    # if before:
    #     params["before"] = before
    # if after:
    #     params["after"] = after
    # if around:
    #     params["around"] = around

    endpoint = f"channels/{channel_id}/messages"
    result = await _make_discord_request("GET", endpoint, params=params)

    if result["status"] == "success":
        return {"status": "success", "messages": result["data"]}
    return result