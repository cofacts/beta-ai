"""
Fact-checking tools for Cofacts AI agents to verify suspicious messages and claims.

Articles in Cofacts represent suspicious messages reported by users through LINE.
Each Article may have multiple ArticleReplies (fact-check responses from collaborators)
and ReplyRequests (additional context provided by reporters or collaborators).
"""

import json
import asyncio
from typing import Dict, List, Any, Optional
import httpx


async def search_cofacts_database(
    query: Optional[str] = None,
    article_ids: Optional[List[str]] = None,
    limit: int = 10,
    after: Optional[str] = None,
    has_article_reply_with_more_positive_feedback: Optional[bool] = None,
    reply_count_max: Optional[int] = None,
    days_back: Optional[int] = None,
    order_by: str = "_score"
) -> Dict[str, Any]:
    """
    Search the Cofacts database for articles using various filters.

    This unified function can:
    - Search by text similarity (query parameter)
    - Get specific articles by IDs (article_ids parameter)
    - Find trending articles needing fact-checks (reply_count_max + days_back)
    - Apply various filters and sorting options

    Cofacts Articles represent suspicious messages reported by LINE users. Key information includes:
    - articleType: Whether the message is TEXT, IMAGE, VIDEO, or AUDIO
    - text: For text messages, this is the content. For media, this is OCR/transcript result
    - attachmentUrl: Preview of media content (when articleType is not TEXT)
    - articleReplies: Fact-check responses from collaborators with feedback scores
    - replyRequests: Additional context from reporters with community ratings
    - hyperlinks: URLs found in the message with crawled metadata
    - cooccurrences: Messages reported together, indicating they were shared as a set
    - relatedArticles: Similar messages that may have existing fact-checks

    Args:
        query: The suspicious message or claim to search for (for similarity search)
        article_ids: List of specific article IDs to retrieve (alternative to query)
        limit: Maximum number of results to return (default: 10)
        after: Cursor for pagination - returns results after this cursor
        has_article_reply_with_more_positive_feedback: Filter for articles with well-received replies
        reply_count_max: Maximum number of replies (useful for finding articles that need more fact-checks)
        days_back: Only include articles created within this many days (useful for trending articles)
        order_by: Sort order - "_score" (relevance), "replyRequestCount" (trending), "createdAt"

    Returns:
        Search results from Cofacts database with pagination info
    """
    try:
        # Build filter object based on parameters
        filter_obj = {}

        if query:
            filter_obj["moreLikeThis"] = {
                "like": query,
                "minimumShouldMatch": "0"
            }

        if article_ids:
            filter_obj["ids"] = article_ids

        if has_article_reply_with_more_positive_feedback is not None:
            filter_obj["hasArticleReplyWithMorePositiveFeedback"] = has_article_reply_with_more_positive_feedback

        if reply_count_max is not None:
            filter_obj["replyCount"] = {"LT": reply_count_max}

        if days_back is not None:
            from datetime import datetime, timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            filter_obj["createdAt"] = {
                "GTE": start_date.isoformat(),
                "LTE": end_date.isoformat()
            }

        # Build orderBy based on order_by parameter
        if order_by == "replyRequestCount":
            order_by_obj = [{"replyRequestCount": "DESC"}, {"createdAt": "DESC"}]
        elif order_by == "createdAt":
            order_by_obj = [{"createdAt": "DESC"}]
        else:  # default to _score
            order_by_obj = [{"_score": "DESC"}]

        graphql_query = """
        query ListArticles($filter: ListArticleFilter!, $orderBy: [ListArticleOrderBy!]!, $first: Int!, $after: String) {
          ListArticles(
            filter: $filter
            orderBy: $orderBy
            first: $first
            after: $after
          ) {
            totalCount
            pageInfo {
              firstCursor
              lastCursor
            }
            edges {
              node {
                id
                text
                createdAt
                articleType
                attachmentUrl(variant: PREVIEW)
                replyCount
                replyRequestCount
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
                  positiveFeedbackCount
                  negativeFeedbackCount
                  feedbacks(statuses: [NORMAL]) {
                    vote
                    comment
                    createdAt
                    user {
                      name
                    }
                  }
                }
                replyRequests(statuses: [NORMAL]) {
                  user {
                    name
                  }
                  reason
                  createdAt
                  positiveFeedbackCount
                  negativeFeedbackCount
                }
                hyperlinks {
                  url
                  normalizedUrl
                  title
                  summary
                  topImageUrl
                  status
                  error
                }
                cooccurrences {
                  articleIds
                  createdAt
                }
                relatedArticles(first: 5) {
                  edges {
                    node {
                      id
                      text
                      articleType
                      replyCount
                    }
                    score
                  }
                }
              }
              score
              cursor
            }
          }
        }
        """

        variables = {
            "filter": filter_obj,
            "orderBy": order_by_obj,
            "first": limit,
            "after": after
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
                "search_params": {
                    "query": query,
                    "variables": variables,
                    "article_ids": article_ids,
                    "order_by": order_by,
                    "reply_count_max": reply_count_max,
                    "days_back": days_back
                },
                "total_count": result["data"]["ListArticles"]["totalCount"],
                "articles": [edge["node"] for edge in result["data"]["ListArticles"]["edges"]],
                "scores": [edge["score"] for edge in result["data"]["ListArticles"]["edges"]],
                "page_info": result["data"]["ListArticles"]["pageInfo"],
                "cursors": [edge["cursor"] for edge in result["data"]["ListArticles"]["edges"]]
            }

    except Exception as e:
        return {
            "error": f"Failed to search Cofacts database: {str(e)}",
            "search_params": {"query": query, "article_ids": article_ids}
        }


async def search_external_factcheck_databases(
    query: str,
    language_code: str = "zh-TW",
    limit: int = 10
) -> Dict[str, Any]:
    """
    Search external fact-checking databases using Google Fact Check Tools API.

    Note: This requires a Google Cloud API key with Fact Check Tools API enabled.
    Currently returns a placeholder response as authentication is not implemented.

    Args:
        query: The claim to fact-check
        language_code: Language code for the search (e.g., "zh-TW", "en")
        limit: Maximum number of results to return

    Returns:
        Fact-check results from Google Fact Check Tools API
    """
    try:
        # Note: Authentication not implemented yet
        # Would need actual Google Cloud API key here
        return {
            "message": "Google Fact Check Tools API requires authentication setup",
            "query": query,
            "language_code": language_code,
            "limit": limit,
            "note": "Use search_cofacts_database for available fact-checks in Traditional Chinese"
        }

        # Placeholder for future implementation:
        # api_key = "YOUR_GOOGLE_FACTCHECK_API_KEY"
        # params = {
        #     "query": query,
        #     "languageCode": language_code,
        #     "pageSize": limit,
        #     "key": api_key
        # }
        #
        # async with httpx.AsyncClient(timeout=30.0) as client:
        #     response = await client.get(
        #         "https://factchecktools.googleapis.com/v1alpha1/claims:search",
        #         params=params
        #     )
        #     response.raise_for_status()
        #     result = response.json()
        #     return {
        #         "query": query,
        #         "language_code": language_code,
        #         "claims": result.get("claims", []),
        #         "next_page_token": result.get("nextPageToken"),
        #         "total_results": len(result.get("claims", []))
        #     }

    except Exception as e:
        return {
            "error": f"Failed to search Google Fact Check Tools API: {str(e)}",
            "query": query,
            "language_code": language_code
        }


async def search_specific_cofacts_article(
    article_id: str
) -> Dict[str, Any]:
    """
    Get a specific article from Cofacts database by ID.

    Returns detailed information about a Cofacts article including:
    - Full text content or OCR/transcript for media
    - All fact-check responses (articleReplies) with community feedback
    - Additional context from reporters (replyRequests) with ratings
    - Related articles that might have existing fact-checks
    - Cooccurrences showing messages shared together
    - URL metadata for any links in the message

    The article ID can be used to construct Cofacts URLs: https://cofacts.tw/article/{article_id}

    Args:
        article_id: The Cofacts article ID to retrieve

    Returns:
        Detailed article information from Cofacts
    """
    try:
        graphql_query = """
        query GetArticle($id: String!) {
          GetArticle(id: $id) {
            id
            text
            createdAt
            articleType
            attachmentUrl(variant: PREVIEW)
            attachmentHash
            replyCount
            replyRequestCount
            contributors {
              user {
                name
              }
              updatedAt
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
                hyperlinks {
                  url
                  normalizedUrl
                  title
                  summary
                  topImageUrl
                  status
                  error
                }
              }
              user {
                name
              }
              createdAt
              positiveFeedbackCount
              negativeFeedbackCount
              feedbacks(statuses: [NORMAL]) {
                vote
                comment
                createdAt
                user {
                  name
                }
              }
            }
            replyRequests(statuses: [NORMAL]) {
              user {
                name
              }
              reason
              createdAt
              positiveFeedbackCount
              negativeFeedbackCount
            }
            hyperlinks {
              url
              normalizedUrl
              title
              summary
              topImageUrl
              status
              error
            }
            cooccurrences {
              id
              articleIds
              createdAt
              articles {
                id
                text
                articleType
                attachmentUrl(variant: PREVIEW)
              }
            }
            relatedArticles(first: 10) {
              totalCount
              edges {
                node {
                  id
                  text
                  articleType
                  replyCount
                  createdAt
                  articleReplies(statuses: [NORMAL]) {
                    reply {
                      id
                      type
                      text
                    }
                    positiveFeedbackCount
                    negativeFeedbackCount
                  }
                }
                score
              }
            }
            stats(dateRange: { GTE: "now-90d/d" }) {
              date
              lineUser
              lineVisit
              webUser
              webVisit
              liffUser
              liffVisit
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
    reference: str
) -> Dict[str, Any]:
    """
    Submit a fact-check reply to Cofacts (requires authentication).

    Note: This requires authentication with Cofacts API which is not yet implemented.
    Currently returns a placeholder response.

    Args:
        article_id: The Cofacts article ID to reply to
        reply_type: Type of reply ("RUMOR", "NOT_RUMOR", "OPINIONATED", "NOT_ARTICLE")
        text: The fact-check response text
        reference: URLs and summaries as references

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





