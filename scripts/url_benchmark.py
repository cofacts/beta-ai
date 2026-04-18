import asyncio
import os
import datetime
import argparse
import httpx
import difflib
import json
from bs4 import BeautifulSoup
from langfuse import Langfuse
from typing import Dict, Any

# ============================================================================
# 1. 初始化設定
# ============================================================================
DATASET_NAME = "URL-Extraction-Benchmark"

# 初始化 Langfuse Client (會自動讀取環境變數 LANGFUSE_PUBLIC_KEY 等)
langfuse = Langfuse()

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
    async def extract_unfurl(url: str) -> Dict[str, Any]:
        """
        技術 A: 輕量級 Unfurl / Metadata 解析
        """
        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                response = await client.get(url)
                soup = BeautifulSoup(response.content, "html.parser")

                title = ""
                if soup.title:
                    title = soup.title.string
                og_title = soup.find("meta", property="og:title")
                if og_title and og_title.get("content"):
                    title = og_title.get("content")

                summary = ""
                description = soup.find("meta", attrs={"name": "description"})
                if description and description.get("content"):
                    summary = description.get("content")
                og_description = soup.find("meta", property="og:description")
                if og_description and og_description.get("content"):
                    summary = og_description.get("content")

                top_image_url = ""
                og_image = soup.find("meta", property="og:image")
                if og_image and og_image.get("content"):
                    top_image_url = og_image.get("content")

                return {
                    "method": "unfurl",
                    "url": url,
                    "title": title or "",
                    "summary": summary or "",
                    "topImageUrl": top_image_url or "",
                    "status": response.status_code,
                    "error": None
                }
        except Exception as e:
            return {
                "method": "unfurl",
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
        技術 C: Gemini 原生 URL Context Tool
        """
        import litellm

        prompt = f"""
Please read the following URL and extract its main content.
Return ONLY a JSON object with the following keys:
- "title": The title of the page
- "summary": A brief summary or transcription of the main content
- "topImageUrl": The main image URL if one exists, otherwise an empty string

URL: {url}
"""

        try:
            response = await litellm.acompletion(
                model="gemini/gemini-2.5-pro",
                messages=[{"role": "user", "content": prompt}],
                api_key=os.environ.get("GEMINI_API_KEY"),
                response_format={"type": "json_object"}
            )

            result_text = response.choices[0].message.content
            parsed = json.loads(result_text)

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
        "Tech-A-Unfurl": Extractors.extract_unfurl,
        "Tech-B-Legacy-Resolver": Extractors.extract_legacy_url_resolver,
        "Tech-C-Gemini-URL": Extractors.extract_gemini_context,
        "Tech-D-Agent-Browser": Extractors.extract_agent_browser,
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
            run_name = f"{tech_name}-{custom_run_name}-{timestamp}"
        else:
            run_name = f"{tech_name}-Run-{timestamp}"

        for item in dataset.items:
            target_url = item.input.get("url")
            if not target_url:
                continue

            print(f"  🔍 解析中: {target_url}")

            start_time = datetime.datetime.now()
            output_data = None
            error_data = None

            try:
                # 執行解析
                output_data = await extractor_fn(target_url)
            except Exception as e:
                print(f"  ❌ 解析失敗: {str(e)}")
                error_data = str(e)

            end_time = datetime.datetime.now()

            # 建立 Langfuse Generation 紀錄
            generation = langfuse.generation(
                name=f"Extract-{tech_name}",
                model=tech_name, # 使用 model 欄位來區分不同技術，方便在 UI 比較
                input={"url": target_url},
                output={"error": error_data} if error_data else output_data,
                start_time=start_time,
                end_time=end_time,
            )

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

                # 3. 將分數送回 Langfuse 綁定至該 Generation
                scores = [
                    ("title_similarity", title_score),
                    ("summary_similarity", summary_score),
                    ("image_similarity", img_score),
                    ("overall_similarity", overall_score)
                ]

                for score_name, score_value in scores:
                    langfuse.score(
                        trace_id=generation.trace_id,
                        observation_id=generation.id,
                        name=score_name,
                        value=score_value
                    )
            # ====================================================================

            # 將 Generation 連結至 Dataset Item
            item.link(
                generation,
                run_name=run_name
            )

        print(f"✅ [{tech_name}] 執行完畢！\n")

    # 確保所有事件與分數都送出至 Langfuse
    langfuse.flush()
    print("🎉 所有的 Benchmark 數據與 Scores 皆已上傳至 Langfuse！請至 UI 查看並排比較結果。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="執行 Cofacts URL Extraction Benchmark 腳本")
    parser.add_argument(
        "-m", "--method",
        type=str,
        choices=[
            "Tech-A-Unfurl",
            "Tech-B-Legacy-Resolver",
            "Tech-C-Gemini-URL",
            "Tech-D-Agent-Browser"
        ],
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
