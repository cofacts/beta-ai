import asyncio
import os
import datetime
import argparse
import httpx
import difflib
import json
from bs4 import BeautifulSoup
from langfuse import get_client, Langfuse
from typing import Dict, Any

# ============================================================================
# 1. 初始化設定
# ============================================================================
DATASET_NAME = "urls"

# 初始化 Langfuse Client (會自動讀取環境變數 LANGFUSE_PUBLIC_KEY 等)
langfuse = get_client()

# ============================================================================
# 工具函式：計算字串相似度
# ============================================================================
def calculate_similarity(expected: str, actual: str) -> float:
    """計算兩個字串的相似度 (0.0 ~ 1.0)"""
    e_str = str(expected or "").strip()
    a_str = str(actual or "").strip()

    if not e_str and not a_str:
        return 1.0  # 兩者皆為空，視為完全一致
    if not e_str or not a_str:
        return 0.0  # 一個有值一個為空，完全不一致

    return difflib.SequenceMatcher(None, e_str, a_str).ratio()

# ============================================================================
# 2. 定義各種不同的解析技術 (Extractors)
# ============================================================================
class Extractors:

    @staticmethod
    async def extract_cloudflare_browser(url: str) -> Dict[str, Any]:
        """
        技術 A: Cloudflare Browser Run JSON Endpoint
        """
        try:
            cf_account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
            cf_api_token = os.environ.get("CLOUDFLARE_API_TOKEN")

            if not cf_account_id or not cf_api_token:
                return {
                    "method": "cloudflare-browser-json",
                    "url": url,
                    "title": "",
                    "summary": "",
                    "topImageUrl": "",
                    "status": 500,
                    "error": "Missing CLOUDFLARE_ACCOUNT_ID or CLOUDFLARE_API_TOKEN"
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
                            "title": { "type": "string" },
                            "summary": { "type": "string" },
                            "topImageUrl": { "type": "string" }
                        },
                        "required": ["title", "summary"]
                    }
                }
            }

            headers = {
                "authorization": f"Bearer {cf_api_token}",
                "content-type": "application/json"
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(endpoint, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()

                if data.get("success"):
                    result = data.get("result", {})
                    return {
                        "method": "cloudflare-browser-json",
                        "url": url,
                        "title": result.get("title", ""),
                        "summary": result.get("summary", ""),
                        "topImageUrl": result.get("topImageUrl", ""),
                        "status": 200,
                        "error": None
                    }
                else:
                    return {
                        "method": "cloudflare-browser-json",
                        "url": url,
                        "title": "",
                        "summary": "",
                        "topImageUrl": "",
                        "status": response.status_code,
                        "error": str(data.get("errors", "Unknown error"))
                    }
        except Exception as e:
            return {
                "method": "cloudflare-browser-json",
                "url": url,
                "title": "",
                "summary": "",
                "topImageUrl": "",
                "status": 500,
                "error": str(e)
            }

    @staticmethod
    async def extract_legacy_url_resolver(url: str) -> Dict[str, Any]:
        """
        技術 B: 舊版 url-resolver (Baseline)
        """
        import grpc
        import src.typeDefs.url_resolver_pb2 as url_resolver_pb2
        import src.typeDefs.url_resolver_pb2_grpc as url_resolver_pb2_grpc

        channel = grpc.aio.insecure_channel('localhost:4000')
        stub = url_resolver_pb2_grpc.UrlResolverStub(channel)

        request = url_resolver_pb2.UrlsRequest(urls=[url])

        try:
            async for reply in stub.ResolveUrl(request):
                return {
                    "method": "legacy-url-resolver",
                    "url": reply.url,
                    "title": reply.title,
                    "summary": reply.summary,
                    "topImageUrl": reply.top_image_url,
                    "status": reply.status,
                    "error": None if reply.HasField('successfully_resolved') else str(reply.error)
                }
        except Exception as e:
            return {
                "method": "legacy-url-resolver",
                "url": url,
                "title": "",
                "summary": "",
                "topImageUrl": "",
                "status": 500,
                "error": str(e)
            }
        finally:
            await channel.close()

        return {
            "method": "legacy-url-resolver",
            "url": url,
            "title": "",
            "summary": "",
            "topImageUrl": "",
            "status": 500,
            "error": "No reply from server"
        }

    @staticmethod
    async def extract_gemini_context(url: str) -> Dict[str, Any]:
        """
        技術 C: Gemini 原生 URL Context Tool (使用 google-genai)
        """
        from google import genai
        from google.genai import types
        import os
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

            # 移除可能的 markdown code block
            result_text = re.sub(r"^```(?:json)?\s*", "", result_text)
            result_text = re.sub(r"\s*```$", "", result_text)

            parsed = json.loads(result_text.strip())

            return {
                "method": "gemini-url-context",
                "url": url,
                "title": parsed.get("title", ""),
                "summary": parsed.get("summary", ""),
                "topImageUrl": parsed.get("topImageUrl", ""),
                "status": 200,
                "error": None
            }
        except Exception as e:
            return {
                "method": "gemini-url-context",
                "url": url,
                "title": "",
                "summary": "",
                "topImageUrl": "",
                "status": 500,
                "error": str(e)
            }

    @staticmethod
    async def extract_agent_browser(url: str) -> Dict[str, Any]:
        """
        技術 D: Agentic Browser Rendering (搭配 ADK Computer Use / Playwright)
        """
        # 呼叫 resolve_url_agent 的 run 方法來取得資料
        from resolve_url_agent.agent import resolve_url_agent

        try:
            # 讓 agent 處理這個網址，要求只回傳 JSON 格式
            result = await resolve_url_agent.run(
                f"Please navigate to this url: {url} and extract its title, summary, and topImageUrl as requested in your instructions. Return ONLY a JSON object.",
            )

            result_text = result.message.content

            # Agent 有時可能會回傳 ```json ... ```，我們需要清掉
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]

            parsed = json.loads(result_text.strip())

            return {
                "method": "agentic-browser",
                "url": url,
                "title": parsed.get("title", ""),
                "summary": parsed.get("summary", ""),
                "topImageUrl": parsed.get("topImageUrl", ""),
                "status": 200,
                "error": None
            }
        except Exception as e:
            return {
                "method": "agentic-browser",
                "url": url,
                "title": "",
                "summary": "",
                "topImageUrl": "",
                "status": 500,
                "error": str(e)
            }

# ============================================================================
# 3. 核心 Benchmark 執行邏輯
# ============================================================================
async def run_benchmark(selected_method: str = None, custom_run_name: str = None):
    print(f"[1] 正在連接 Langfuse 並取得 Dataset: {DATASET_NAME}...")

    try:
        dataset = langfuse.get_dataset(DATASET_NAME)
    except Exception as e:
        print(f"❌ 無法取得 Dataset，請確保 Langfuse 上已建立名為 {DATASET_NAME} 的 Dataset。")
        print(f"錯誤訊息: {e}")
        return

    print(f"✅ 成功取得 Dataset，共 {len(dataset.items)} 筆測試 URL。\n")

    # 將方法對應到執行函式
    all_methods = {
        "cf-browser": Extractors.extract_cloudflare_browser,
        "url-resolver": Extractors.extract_legacy_url_resolver,
        "url-context": Extractors.extract_gemini_context,
        "computer-use": Extractors.extract_agent_browser,
    }

    # 如果有指定特定的方法，過濾出該方法；否則跑全部
    methods_to_run = all_methods
    if selected_method:
        if selected_method not in all_methods:
            print(f"❌ 找不到指定的方法: {selected_method}")
            return
        methods_to_run = {selected_method: all_methods[selected_method]}

    # 開始依序執行所選的技術
    for tech_name, extractor_fn in methods_to_run.items():
        print(f"🚀 開始執行技術評測：[{tech_name}]")

        # 建立這次 Run 的唯一名稱
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if custom_run_name:
            run_name = f"{custom_run_name}-{timestamp}"
        else:
            run_name = f"{tech_name}-{timestamp}"

        for item in dataset.items:
            # Handle both dict and string inputs from Langfuse dataset
            input_val = item.input
            target_url = input_val.get("url") if isinstance(input_val, dict) else input_val

            if not target_url:
                continue

            print(f"  🔍 解析中: {target_url}")

            # 使用 item.run() context manager 自動建立 trace 並連結至 dataset item
            with item.run(
                run_name=run_name,
                run_description=f"{tech_name}",
                run_metadata={"tech": tech_name},
            ) as root_span:
                output_data = None
                error_data = None

                # 建立 Generation observation 紀錄
                with langfuse.start_as_current_observation(
                    as_type="generation",
                    name=f"Extract-{tech_name}",
                    model=tech_name,
                    input={"url": target_url},
                ) as generation:
                    try:
                        # 執行解析
                        output_data = await extractor_fn(target_url)
                        generation.update(output=output_data)
                    except Exception as e:
                        print(f"  ❌ 解析失敗: {str(e)}")
                        error_data = str(e)
                        generation.update(output={"error": error_data})

                # ====================================================================
                # 計算並上傳 Similarity Score (如果 Dataset 有預期結果)
                # ====================================================================
                expected = item.expected_output
                if output_data and not error_data and expected and isinstance(expected, dict):
                    # 1. 分別計算 3 個欄位的分數
                    title_score = calculate_similarity(expected.get("title"), output_data.get("title"))
                    summary_score = calculate_similarity(expected.get("summary"), output_data.get("summary"))
                    img_score = calculate_similarity(expected.get("topImageUrl"), output_data.get("topImageUrl"))

                    # 2. 計算綜合評分 (此處以等權重算術平均為例)
                    overall_score = (title_score + summary_score + img_score) / 3.0

                    # 3. 將分數送回 Langfuse 綁定至該 trace
                    scores = [
                        ("title_similarity", title_score),
                        ("summary_similarity", summary_score),
                        ("image_similarity", img_score),
                        ("overall_similarity", overall_score)
                    ]

                    for score_name, score_value in scores:
                        root_span.score_trace(
                            name=score_name,
                            value=score_value,
                        )
                # ====================================================================

        print(f"✅ [{tech_name}] 執行完畢！\n")

    # 確保所有事件與分數都送出至 Langfuse
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

    # 使用 asyncio 執行主程式，並傳入 CLI 解析後的參數
    asyncio.run(run_benchmark(selected_method=args.method, custom_run_name=args.run_name))
