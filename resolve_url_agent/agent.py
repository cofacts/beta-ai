import base64
import os

from google.adk import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.tools.computer_use.computer_use_toolset import ComputerUseToolset
from google.genai import types as genai_types

from .playwright import PlaywrightComputer

cf_account_id = os.environ.get('CLOUDFLARE_ACCOUNT_ID')
cf_api_token = os.environ.get('CLOUDFLARE_API_TOKEN')

cdp_url = None
cdp_headers = None

if cf_account_id and cf_api_token:
    cdp_url = f'wss://api.cloudflare.com/client/v4/accounts/{cf_account_id}/browser-rendering/devtools/browser?keep_alive=60000'
    cdp_headers = {'Authorization': f'Bearer {cf_api_token}'}


_AGENT_INSTRUCTION = '''You are an agent that reads a webpage that has already been loaded in your browser.

ABSOLUTE RULES — violating these wastes the entire run:
1. The page is ALREADY open at the target URL. Do NOT call open_web_browser, do NOT navigate, do NOT type URLs.
2. Take ONE look at the current screen and produce the JSON answer immediately.
3. Do NOT scroll, click, hover, or wait. The information you need (title, headline, og:image, lead paragraph) is in the visible viewport or page metadata.
4. If the page seems blank or is still loading, return whatever you can extract — even partial answers are better than running more tool calls.

Return ONLY this JSON object, with no prose, no code fences, no commentary:
{
  "title": "<page title or main headline>",
  "summary": "<one-paragraph summary of the visible main content>",
  "topImageUrl": "<the main image URL (og:image-style hero image), or empty string>"
}
'''


def _rewrite_screenshots_to_inline_data(
    callback_context: CallbackContext, llm_request: LlmRequest
):
    """Move screenshot bytes from dict payload to FunctionResponse.parts.

    ADK's ComputerUseTool.run_async returns ``{"image": {"data": <base64>}, "url": ...}``.
    ADK's flow then calls ``Part.from_function_response(response=<that dict>)``,
    embedding the base64 string inside the structured response field. Gemini's
    Computer Use model can read it that way, but the bytes are billed as text
    tokens — multi-turn runs blow past the 131k input limit on the second or
    third tool call.

    Gemini's native Computer Use protocol (per the official cookbook) attaches
    screenshots as ``FunctionResponse.parts = [FunctionResponsePart(inline_data=FunctionResponseBlob(...))]``.
    Bytes attached this way are processed through the multimodal pipeline —
    each image costs roughly ``258 * tile_count`` tokens regardless of file
    size, an order of magnitude cheaper than base64 text.

    This callback runs right before every Gemini call, so we rewrite history
    in place:
      - Last screenshot: convert to ``inline_data`` so the model still sees
        the current viewport, billed cheaply.
      - Older screenshots: drop the bytes entirely (keep an empty marker)
        so the model knows what was tried but pays no token cost.
    """
    contents = llm_request.contents or []
    found: list[tuple[genai_types.FunctionResponse, dict]] = []

    for content in contents:
        for part in content.parts or []:
            fr = getattr(part, 'function_response', None)
            if fr is None:
                continue
            resp = getattr(fr, 'response', None)
            if not isinstance(resp, dict):
                continue
            image = resp.get('image')
            if isinstance(image, dict) and image.get('data'):
                found.append((fr, image))

    if not found:
        return None

    for fr, image in found[:-1]:
        image['data'] = ''

    last_fr, last_image = found[-1]
    raw = last_image['data']
    try:
        screenshot_bytes = base64.b64decode(raw) if isinstance(raw, str) else raw
    except Exception:
        return None

    last_fr.parts = [
        genai_types.FunctionResponsePart(
            inline_data=genai_types.FunctionResponseBlob(
                mime_type='image/png',
                data=screenshot_bytes,
            )
        )
    ]
    last_image['data'] = ''

    return None


def build_resolve_url_agent(initial_url: str = 'https://www.google.com') -> Agent:
    return Agent(
        model='gemini-2.5-computer-use-preview-10-2025',
        name='resolve_url_agent',
        description='computer use agent that can operate a browser on a computer to read URLs and extract their content',
        instruction=_AGENT_INSTRUCTION,
        before_model_callback=_rewrite_screenshots_to_inline_data,
        tools=[
            ComputerUseToolset(computer=PlaywrightComputer(
                screen_size=(1024, 768),
                initial_url=initial_url,
                cdp_url=cdp_url,
                cdp_headers=cdp_headers,
            ))
        ],
    )


resolve_url_agent = build_resolve_url_agent()
