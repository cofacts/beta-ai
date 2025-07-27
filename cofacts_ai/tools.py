"""
Fact-checking tools for Cofacts AI agents to verify suspicious messages and claims.
"""

import json
import asyncio
from typing import Dict, List, Any, Optional
import httpx

from google.adk.tools.tool_context import ToolContext


async def search_cofacts_database(
    query: str,
    tool_context: ToolContext,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Search the Cofacts database for existing fact-checks using GraphQL API.

    Args:
        query: The suspicious message or claim to search for
        tool_context: Tool context for the agent
        limit: Maximum number of results to return

    Returns:
        Search results from Cofacts database
    """
    try:
        graphql_query = """
        query ListArticles($moreLikeThis: MoreLikeThisInput!, $first: Int!) {
          ListArticles(
            filter: {
              moreLikeThis: $moreLikeThis
              hasArticleReplyWithMorePositiveFeedback: true
            }
            orderBy: [{ _score: DESC }]
            first: $first
          ) {
            totalCount
            edges {
              node {
                id
                text
                createdAt
                updatedAt
                replyCount
                articleType
                attachmentUrl
                articleReplies(statuses: [NORMAL]) {
                  reply {
                    id
                    type
                    text
                    createdAt
                  }
                  user {
                    name
                  }
                  createdAt
                  positiveFeedbackCount
                  negativeFeedbackCount
                }
                articleCategories(statuses: [NORMAL]) {
                  category {
                    title
                    description
                  }
                  positiveFeedbackCount
                  negativeFeedbackCount
                }
                hyperlinks {
                  url
                  title
                  summary
                }
              }
              score
            }
          }
        }
        """

        variables = {
            "moreLikeThis": {
                "like": query,
                "minimumShouldMatch": "0"
            },
            "first": limit
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.cofacts.tw/graphql",
                json={
                    "query": graphql_query,
                    "variables": variables
                },
                headers={
                    "Content-Type": "application/json"
                }
            )
            response.raise_for_status()

            result = response.json()

            if "errors" in result:
                return {
                    "error": f"GraphQL errors: {result['errors']}",
                    "query": query
                }

            return {
                "query": query,
                "total_count": result["data"]["ListArticles"]["totalCount"],
                "articles": [edge["node"] for edge in result["data"]["ListArticles"]["edges"]],
                "scores": [edge["score"] for edge in result["data"]["ListArticles"]["edges"]]
            }

    except Exception as e:
        return {
            "error": f"Failed to search Cofacts database: {str(e)}",
            "query": query
        }


async def search_external_factcheck_databases(
    query: str,
    tool_context: ToolContext,
    language_code: str = "zh-TW",
    limit: int = 10
) -> Dict[str, Any]:
    """
    Search external fact-checking databases using Google Fact Check Tools API.

    Args:
        query: The claim to fact-check
        tool_context: Tool context for the agent
        language_code: Language code for the search (e.g., "zh-TW", "en")
        limit: Maximum number of results to return

    Returns:
        Fact-check results from Google Fact Check Tools API
    """
    try:
        # Note: You'll need to get an API key from Google Cloud Console
        # and enable the Fact Check Tools API
        api_key = "YOUR_GOOGLE_FACTCHECK_API_KEY"  # Replace with actual API key

        params = {
            "query": query,
            "languageCode": language_code,
            "pageSize": limit,
            "key": api_key
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                "https://factchecktools.googleapis.com/v1alpha1/claims:search",
                params=params
            )
            response.raise_for_status()

            result = response.json()

            return {
                "query": query,
                "language_code": language_code,
                "claims": result.get("claims", []),
                "next_page_token": result.get("nextPageToken"),
                "total_results": len(result.get("claims", []))
            }

    except Exception as e:
        return {
            "error": f"Failed to search Google Fact Check Tools API: {str(e)}",
            "query": query,
            "language_code": language_code
        }


async def search_specific_cofacts_article(
    article_id: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Get a specific article from Cofacts database by ID.

    Args:
        article_id: The Cofacts article ID to retrieve
        tool_context: Tool context for the agent

    Returns:
        Detailed article information from Cofacts
    """
    try:
        graphql_query = """
        query GetArticle($id: ID!) {
          GetArticle(id: $id) {
            id
            text
            createdAt
            updatedAt
            replyCount
            articleType
            attachmentUrl
            attachmentHash
            transcribedAt
            contributors {
              user {
                name
              }
              contributedAt
            }
            user {
              name
            }
            articleReplies(statuses: [NORMAL]) {
              id
              reply {
                id
                type
                text
                createdAt
                reference
                user {
                  name
                }
              }
              user {
                name
              }
              createdAt
              updatedAt
              positiveFeedbackCount
              negativeFeedbackCount
            }
            articleCategories(statuses: [NORMAL]) {
              id
              category {
                id
                title
                description
              }
              user {
                name
              }
              positiveFeedbackCount
              negativeFeedbackCount
              createdAt
            }
            hyperlinks {
              url
              title
              summary
              topImageUrl
            }
            replyRequests(statuses: [NORMAL]) {
              user {
                name
              }
              reason
              createdAt
            }
          }
        }
        """

        variables = {"id": article_id}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.cofacts.tw/graphql",
                json={
                    "query": graphql_query,
                    "variables": variables
                },
                headers={
                    "Content-Type": "application/json"
                }
            )
            response.raise_for_status()

            result = response.json()

            if "errors" in result:
                return {
                    "error": f"GraphQL errors: {result['errors']}",
                    "article_id": article_id
                }

            article = result["data"]["GetArticle"]
            if not article:
                return {
                    "error": f"Article not found",
                    "article_id": article_id
                }

            return {
                "article_id": article_id,
                "article": article
            }

    except Exception as e:
        return {
            "error": f"Failed to get Cofacts article: {str(e)}",
            "article_id": article_id
        }


async def submit_cofacts_reply(
    article_id: str,
    reply_type: str,
    text: str,
    reference: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Submit a fact-check reply to Cofacts (requires authentication).

    Args:
        article_id: The Cofacts article ID to reply to
        reply_type: Type of reply ("RUMOR", "NOT_RUMOR", "OPINIONATED", "NOT_ARTICLE")
        text: The fact-check response text
        reference: URLs and summaries as references
        tool_context: Tool context for the agent

    Returns:
        Result of the submission
    """
    try:
        # Note: This requires authentication with Cofacts API
        # You'll need to implement proper OAuth or API key authentication

        graphql_mutation = """
        mutation CreateReply($text: String!, $type: ReplyTypeEnum!, $reference: String!) {
          CreateReply(text: $text, type: $type, reference: $reference) {
            id
            text
            type
            reference
            createdAt
          }
        }
        """

        variables = {
            "text": text,
            "type": reply_type,
            "reference": reference
        }

        # This is a placeholder - you'll need to implement proper authentication
        return {
            "message": "Reply submission requires authentication setup",
            "article_id": article_id,
            "reply_type": reply_type,
            "text_length": len(text),
            "reference_length": len(reference)
        }

    except Exception as e:
        return {
            "error": f"Failed to submit Cofacts reply: {str(e)}",
            "article_id": article_id
        }


async def get_trending_cofacts_articles(
    tool_context: ToolContext,
    days: int = 7,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Get trending articles from Cofacts that need fact-checking.

    Args:
        tool_context: Tool context for the agent
        days: Number of days to look back for trending articles
        limit: Maximum number of results to return

    Returns:
        List of trending articles that need replies
    """
    try:
        from datetime import datetime, timedelta

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        graphql_query = """
        query ListTrendingArticles($createdAt: TimeRangeInput!, $first: Int!) {
          ListArticles(
            filter: {
              createdAt: $createdAt
              replyCount: { LT: 3 }
              hasArticleReplyWithMorePositiveFeedback: false
            }
            orderBy: [{ replyRequestCount: DESC }, { createdAt: DESC }]
            first: $first
          ) {
            totalCount
            edges {
              node {
                id
                text
                createdAt
                replyRequestCount
                replyCount
                articleType
                attachmentUrl
                replyRequests(statuses: [NORMAL]) {
                  reason
                  createdAt
                  user {
                    name
                  }
                }
                user {
                  name
                }
              }
            }
          }
        }
        """

        variables = {
            "createdAt": {
                "GTE": start_date.isoformat(),
                "LTE": end_date.isoformat()
            },
            "first": limit
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.cofacts.tw/graphql",
                json={
                    "query": graphql_query,
                    "variables": variables
                },
                headers={
                    "Content-Type": "application/json"
                }
            )
            response.raise_for_status()

            result = response.json()

            if "errors" in result:
                return {
                    "error": f"GraphQL errors: {result['errors']}",
                    "days": days
                }

            return {
                "days": days,
                "total_count": result["data"]["ListArticles"]["totalCount"],
                "trending_articles": [edge["node"] for edge in result["data"]["ListArticles"]["edges"]]
            }

    except Exception as e:
        return {
            "error": f"Failed to get trending Cofacts articles: {str(e)}",
            "days": days
        }



