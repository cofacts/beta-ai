"""
Cofacts URL Extraction Benchmark

Usage:
    uv run --env-file .env python scripts/url_benchmark.py [options]

Options:
    -m, --method {cf-browser, url-resolver, url-context, computer-use}
        Specify the extraction method to run (default: all)
    -r, --run-name <NAME>
        Custom run name for identification in Langfuse

Methods:
    cf-browser    - Cloudflare Browser Rendering (JSON schema extraction)
    url-resolver  - Legacy gRPC url-resolver (Internal baseline)
    url-context   - Gemini 2.5 Pro native URL Context tool
    computer-use  - ADK Agentic Browser (Playwright / Computer Use)

Environment:
    Requires .env with LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL,
    and GOOGLE_API_KEY (for Gemini) or CLOUDFLARE_ credentials.
"""

import asyncio
import os
import argparse
import httpx
import difflib
import json
from langfuse import get_client, Evaluation
from typing import Dict, Any

# ============================================================================
# 1. 初始化設定
# ============================================================================
DATASET_NAME = "urls"

# 初始化 Langfuse Client (會自動讀取環境變數 LANGFUSE_PUBLIC_KEY 等)
langfuse = get_client()

# ============================================================================
# 工具函式
# ============================================================================
def calculate_similarity(expected: str, actual: str) -> float:
    """計算兩個字串的相似度 (0.0 ~ 1.0)"""
    e_str = str(expected or "").strip()
    a_str = str(actual or "").strip()

    if not e_str and not a_str:
        return 1.0
    if not e_str or not a_str:
        return 0.0

    return difflib.SequenceMatcher(None, e_str, a_str).ratio()


def _get_url(item) -> str:
    """從 dataset item 取出 URL"""
    input_val = item.input
    return input_val.get("url") if isinstance(input_val, dict) else input_val


# ============================================================================
# 2. Task Functions（每個方法一個 top-level async function）
# ============================================================================
async def task_cf_browser(*, item, **kwargs) -> Dict[str, Any]:
    """技術 A: Cloudflare Browser Run JSON Endpoint"""
    url = _get_url(item)
    print(f"  🔍 解析中: {url}")

    cf_account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    cf_api_token = os.environ.get("CLOUDFLARE_API_TOKEN")

    if not cf_account_id or not cf_api_token:
        return {
            "method": "cf-browser", "url": url,
            "title": "", "summary": "", "topImageUrl": "",
            "status": 500, "error": "Missing CLOUDFLARE_ACCOUNT_ID or CLOUDFLARE_API_TOKEN"
        }

    endpoint = f"https://api.cloudflare.com/client/v4/accounts/{cf_account_id}/browser-rendering/json"
    payload = {
        "url": url,
        "prompt": "Extract the main content from this webpage. Provide the title of the page, a brief summary or transcription of the main content, and the main image URL if one exists (like og:image).",
        "response_format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "topImageUrl": {"type": "string"}
                },
                "required": ["title", "summary"]
            }
        }
    }
    headers = {
        "authorization": f"Bearer {cf_api_token}",
        "content-type": "application/json"
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        if data.get("success"):
            result = data.get("result", {})
            return {
                "method": "cf-browser", "url": url,
                "title": result.get("title", ""),
                "summary": result.get("summary", ""),
                "topImageUrl": result.get("topImageUrl", ""),
                "status": 200, "error": None
            }
        else:
            return {
                "method": "cf-browser", "url": url,
                "title": "", "summary": "", "topImageUrl": "",
                "status": response.status_code,
                "error": str(data.get("errors", "Unknown error"))
            }
    except Exception as e:
        return {
            "method": "cf-browser", "url": url,
            "title": "", "summary": "", "topImageUrl": "",
            "status": 500, "error": str(e)
        }


async def task_url_resolver(*, item, **kwargs) -> Dict[str, Any]:
    """技術 B: 舊版 url-resolver (Baseline, gRPC)"""
    url = _get_url(item)
    print(f"  🔍 解析中: {url}")

    import grpc
    import src.typeDefs.url_resolver_pb2 as url_resolver_pb2
    import src.typeDefs.url_resolver_pb2_grpc as url_resolver_pb2_grpc

    channel = grpc.aio.insecure_channel('localhost:4000')
    stub = url_resolver_pb2_grpc.UrlResolverStub(channel)
    request = url_resolver_pb2.UrlsRequest(urls=[url])

    try:
        async for reply in stub.ResolveUrl(request):
            return {
                "method": "url-resolver",
                "url": reply.url,
                "title": reply.title,
                "summary": reply.summary,
                "topImageUrl": reply.top_image_url,
                "status": reply.status,
                "error": None if reply.HasField('successfully_resolved') else str(reply.error)
            }
    except Exception as e:
        return {
            "method": "url-resolver", "url": url,
            "title": "", "summary": "", "topImageUrl": "",
            "status": 500, "error": str(e)
        }
    finally:
        await channel.close()

    return {
        "method": "url-resolver", "url": url,
        "title": "", "summary": "", "topImageUrl": "",
        "status": 500, "error": "No reply from server"
    }


async def task_url_context(*, item, **kwargs) -> Dict[str, Any]:
    """技術 C: Gemini 原生 URL Context Tool"""
    url = _get_url(item)
    print(f"  🔍 解析中: {url}")

    from google import genai
    from google.genai import types
    import re

    prompt = f"""Please read the following URL and extract its main content.
URL: {url}

Return a JSON object with exactly these fields (no markdown, no code block):
{{
  "title": "<page title>",
  "summary": "<brief summary in Traditional Chinese>",
  "topImageUrl": "<URL of main image, or empty string if none>"
}}"""

    try:
        client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(url_context=types.UrlContext())]
            )
        )

        result_text = response.text.strip()
        result_text = re.sub(r"^```(?:json)?\s*", "", result_text)
        result_text = re.sub(r"\s*```$", "", result_text)
        parsed = json.loads(result_text.strip())

        return {
            "method": "url-context", "url": url,
            "title": parsed.get("title", ""),
            "summary": parsed.get("summary", ""),
            "topImageUrl": parsed.get("topImageUrl", ""),
            "status": 200, "error": None
        }
    except Exception as e:
        return {
            "method": "url-context", "url": url,
            "title": "", "summary": "", "topImageUrl": "",
            "status": 500, "error": str(e)
        }


async def task_computer_use(*, item, **kwargs) -> Dict[str, Any]:
    """技術 D: Agentic Browser Rendering (ADK Computer Use / Playwright)"""
    url = _get_url(item)
    print(f"  🔍 解析中: {url}")

    from resolve_url_agent.agent import resolve_url_agent

    try:
        result = await resolve_url_agent.run(
            f"Please navigate to this url: {url} and extract its title, summary, and topImageUrl as requested in your instructions. Return ONLY a JSON object.",
        )

        result_text = result.message.content
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]

        parsed = json.loads(result_text.strip())

        return {
            "method": "computer-use", "url": url,
            "title": parsed.get("title", ""),
            "summary": parsed.get("summary", ""),
            "topImageUrl": parsed.get("topImageUrl", ""),
            "status": 200, "error": None
        }
    except Exception as e:
        return {
            "method": "computer-use", "url": url,
            "title": "", "summary": "", "topImageUrl": "",
            "status": 500, "error": str(e)
        }


# ============================================================================
# 3. Evaluators
# ============================================================================
def _check_expected(expected_output):
    return expected_output and isinstance(expected_output, dict)

def title_evaluator(*, output, expected_output, **kwargs):
    if not _check_expected(expected_output):
        return None
    score = calculate_similarity(expected_output.get("title"), output.get("title"))
    return Evaluation(name="title_similarity", value=score)

def summary_evaluator(*, output, expected_output, **kwargs):
    if not _check_expected(expected_output):
        return None
    score = calculate_similarity(expected_output.get("summary"), output.get("summary"))
    return Evaluation(name="summary_similarity", value=score)

def image_evaluator(*, output, expected_output, **kwargs):
    if not _check_expected(expected_output):
        return None
    score = calculate_similarity(expected_output.get("topImageUrl"), output.get("topImageUrl"))
    return Evaluation(name="image_similarity", value=score)


def _make_avg_run_evaluator(score_name: str):
    """建立一個 run-level evaluator，計算指定 score 在所有 items 的平均"""
    def evaluator(*, item_results, **kwargs):
        scores = [
            e.value for r in item_results
            for e in r.evaluations
            if e.name == score_name and e.value is not None
        ]
        if not scores:
            return Evaluation(name=f"avg_{score_name}", value=None)
        avg = sum(scores) / len(scores)
        return Evaluation(name=f"avg_{score_name}", value=avg, comment=f"{avg:.2%}")
    evaluator.__name__ = f"avg_{score_name}"
    return evaluator

avg_title = _make_avg_run_evaluator("title_similarity")
avg_summary = _make_avg_run_evaluator("summary_similarity")
avg_image = _make_avg_run_evaluator("image_similarity")


# ============================================================================
# 4. 核心 Benchmark 執行邏輯
# ============================================================================
def run_benchmark(selected_method: str = None, custom_run_name: str = None):
    print(f"[1] 正在連接 Langfuse 並取得 Dataset: {DATASET_NAME}...")

    try:
        dataset = langfuse.get_dataset(DATASET_NAME)
    except Exception as e:
        print(f"❌ 無法取得 Dataset，請確保 Langfuse 上已建立名為 {DATASET_NAME} 的 Dataset。")
        print(f"錯誤訊息: {e}")
        return

    print(f"✅ 成功取得 Dataset，共 {len(dataset.items)} 筆測試 URL。\n")

    all_methods = {
        "cf-browser": task_cf_browser,
        "url-resolver": task_url_resolver,
        "url-context": task_url_context,
        "computer-use": task_computer_use,
    }

    methods_to_run = all_methods
    if selected_method:
        if selected_method not in all_methods:
            print(f"❌ 找不到指定的方法: {selected_method}")
            return
        methods_to_run = {selected_method: all_methods[selected_method]}

    for tech_name, task_fn in methods_to_run.items():
        print(f"🚀 開始執行技術評測：[{tech_name}]")

        run_name = custom_run_name if custom_run_name else tech_name

        result = dataset.run_experiment(
            name=run_name,
            description=tech_name,
            task=task_fn,
            evaluators=[title_evaluator, summary_evaluator, image_evaluator],
            run_evaluators=[avg_title, avg_summary, avg_image],
            metadata={"tech": tech_name},
        )

        print(result.format())
        print(f"✅ [{tech_name}] 執行完畢！\n")

    langfuse.flush()
    print("🎉 所有的 Benchmark 數據與 Scores 皆已上傳至 Langfuse！請至 UI 查看並排比較結果。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="執行 Cofacts URL Extraction Benchmark 腳本")
    parser.add_argument(
        "-m", "--method",
        type=str,
        choices=["cf-browser", "url-resolver", "url-context", "computer-use"],
        metavar="{cf-browser,url-resolver,url-context,computer-use}",
        help="指定要單獨執行的解析方法 (預設為全部執行)"
    )
    parser.add_argument(
        "-r", "--run-name",
        type=str,
        help="自訂的 Run 標籤 (例如: new-prompt-v2)，方便在 Langfuse 上識別差異"
    )

    args = parser.parse_args()
    run_benchmark(selected_method=args.method, custom_run_name=args.run_name)
