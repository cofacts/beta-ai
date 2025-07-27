# Cofacts AI - Multi-Agent Fact-Checking System

A sophisticated multi-agent system built on Google's ADK (Agent Development Kit) for automated fact-checking on the Cofacts platform.

## Architecture

The system uses a hierarchical multi-agent architecture with specialized roles:

### 🎯 AI Writer (Orchestrator)
- **Role**: Main orchestrator that composes fact-check replies
- **Capabilities**: 
  - Coordinates all other agents
  - Composes final Cofacts-formatted replies
  - Interacts with human fact-checkers
  - Makes final editorial decisions

### 🔍 AI Investigator 
- **Role**: Deep research specialist
- **Capabilities**:
  - Searches Cofacts database for existing fact-checks
  - Queries external fact-check databases (Google Fact Check Tools API)
  - Conducts web research for primary sources
  - Collects and organizes evidence

### ✅ AI Verifier
- **Role**: Source verification specialist  
- **Capabilities**:
  - Verifies claims against provided URLs
  - Uses Gemini's URL context tool to analyze web content
  - Cross-references multiple sources
  - Assesses source credibility and relevance

### 🗳️ AI Proof-readers
- **Role**: Political perspective reviewers
- **Types**: Progressive, Conservative, Centrist
- **Capabilities**:
  - Review replies for political bias
  - Ensure fairness across political spectrum
  - Suggest improvements for broader appeal
  - Test messaging effectiveness

## Features

### 🔗 Cofacts Integration
- **GraphQL API**: Direct integration with Cofacts database
- **Search**: Find existing fact-checks using semantic similarity
- **Retrieve**: Get detailed article information
- **Submit**: Post new fact-check replies (requires authentication)
- **Trending**: Monitor articles needing fact-checks

### 🌐 External Sources
- **Google Fact Check Tools API**: Access global fact-checking databases
- **Web Navigation**: Gemini's built-in URL context capabilities
- **Multi-language Support**: Chinese (traditional/simplified) and English

### 📝 Cofacts Reply Format
The system produces replies in the official Cofacts format:

#### For "Contains misinformation" or "Contains true information":
```
Text: Brief intro pointing out correct/incorrect parts
References: 
- URL1 - One-line summary
- URL2 - One-line summary  
```

#### For "Contains personal perspective":
```
Text: (1) Explain opinion parts, (2) Remind audience it's not factual
Opinion Sources:
- URL1 - One-line summary
- URL2 - One-line summary
```

## Setup

### Prerequisites
- Python 3.8+
- Google ADK installed
- API keys for external services

### Installation
```bash
# Install dependencies
pip install google-adk httpx

# Set up API keys (in environment or config)
export GOOGLE_FACTCHECK_API_KEY="your_api_key_here"
```

### Configuration
Update the API key in `tools.py`:
```python
# In search_external_factcheck_databases function
api_key = "YOUR_GOOGLE_FACTCHECK_API_KEY"
```

## Usage

### Basic Usage
```python
from cofacts_ai import cofacts_ai_agent

# Fact-check a suspicious message
response = await cofacts_ai_agent.run_async(
    "Please fact-check this message: [suspicious message text]"
)
```

### Advanced Usage with Sub-agents
```python
from cofacts_ai import (
    ai_investigator, 
    ai_verifier,
    ai_proofreader_progressive
)

# Use investigator for research
research = await ai_investigator.run_async(
    "Research claims about COVID-19 vaccines"
)

# Use verifier for source checking  
verification = await ai_verifier.run_async(
    "Verify this claim against: https://example.com/source"
)

# Get political perspective feedback
feedback = await ai_proofreader_progressive.run_async(
    "Review this fact-check reply: [draft text]"
)
```

### Example Workflow
```python
# 1. Research phase
research_results = await ai_investigator.run_async(
    "Please search Cofacts and external databases for: [claim]"
)

# 2. Verification phase  
verification_results = await ai_verifier.run_async(
    "Please verify these sources support the claims: [sources]"
)

# 3. Composition phase
draft_reply = await ai_writer.run_async(
    f"Compose fact-check reply based on: {research_results} and {verification_results}"
)

# 4. Review phase
progressive_review = await ai_proofreader_progressive.run_async(f"Review: {draft_reply}")
conservative_review = await ai_proofreader_conservative.run_async(f"Review: {draft_reply}")

# 5. Finalization
final_reply = await ai_writer.run_async(
    f"Finalize reply incorporating feedback: {progressive_review}, {conservative_review}"
)
```

## Tools Available

### Cofacts Tools
- `search_cofacts_database(query, limit=10)`: Search for similar fact-checks
- `search_specific_cofacts_article(article_id)`: Get detailed article info
- `get_trending_cofacts_articles(days=7, limit=10)`: Find trending articles needing replies
- `submit_cofacts_reply(article_id, reply_type, text, reference)`: Submit new fact-check

### External Tools  
- `search_external_factcheck_databases(query, language_code="zh-TW")`: Search global fact-check databases

## Development

### Adding New Tools
```python
async def new_fact_check_tool(
    parameter: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Description of what this tool does.
    """
    try:
        # Implementation
        return {"result": "success"}
    except Exception as e:
        return {"error": str(e)}
```

### Customizing Agents
You can modify agent instructions and capabilities by editing `agent.py`:

```python
custom_agent = LlmAgent(
    name="custom_factchecker",
    model="gemini-2.5-pro", 
    description="Custom fact-checker with specific focus",
    instruction="Your custom instructions here...",
    tools=[list_of_tools]
)
```

## Best Practices

### For Fact-Checkers
1. **Start with Research**: Always use ai_investigator first to gather evidence
2. **Verify Sources**: Use ai_verifier to check if sources actually support claims  
3. **Consider Perspectives**: Get feedback from multiple political proof-readers
4. **Be Transparent**: Include reasoning and source quality assessment
5. **Stay Neutral**: Focus on facts, not political implications

### For Developers
1. **Handle Errors**: All tools include error handling and return structured responses
2. **Rate Limiting**: Be mindful of API rate limits for external services
3. **Authentication**: Implement proper auth for Cofacts submission features
4. **Monitoring**: Log agent interactions for quality improvement

## Contributing

1. Fork the repository
2. Create feature branches for new capabilities
3. Add tests for new tools and agents
4. Update documentation
5. Submit pull requests

## License

Licensed under the Apache License, Version 2.0. See LICENSE file for details.
