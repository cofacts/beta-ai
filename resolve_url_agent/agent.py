import json
from google.adk import Agent
from google.adk.models.google_llm import Gemini
from google.adk.tools.computer_use.computer_use_toolset import ComputerUseToolset

from .playwright import PlaywrightComputer

# We'll need playwright implementation from earlier

resolve_url_agent = Agent(
    model='gemini-2.5-computer-use-preview-10-2025',
    name='resolve_url_agent',
    description='computer use agent that can operate a browser on a computer to read URLs and extract their content',
    instruction='''You are an agent that uses a browser to visit a specific URL, reads the webpage, and extracts its main content.
You must summarize the webpage and provide a JSON response with the following keys:
- "title": The title of the page
- "summary": A brief summary or transcription of the main content
- "topImageUrl": The main image URL if one exists (like og:image), otherwise an empty string

Return ONLY a JSON object.
''',
    tools=[
        ComputerUseToolset(computer=PlaywrightComputer(screen_size=(1280, 936)))
    ],
)
