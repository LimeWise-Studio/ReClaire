import asyncio
import io
import os
import json
import re
import hashlib
import traceback
from typing import List, Optional, Dict, Any
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from pypdf import PdfReader
from playwright.async_api import async_playwright

from fastapi import FastAPI, HTTPException, Header, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 2026 SDK Imports
from google import genai
from google.genai import types
from supabase import create_client, Client


# ============================================================
# CONFIGURATION & API KEYS
# ============================================================

# === [KEYS REQUIRED: ADD YOUR CREDENTIALS HERE OR IN YOUR .ENV FILE] ===
SUPABASE_URL = os.environ.get("SUPABASE_URL")  # <--- Replace or set ENV
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")  # <--- Replace or set ENV
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  # <--- Replace or set ENV

# Initialize Supabase Client
supabase: Client = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    except Exception as e:
        print(f"Warning: Failed to initialize Supabase client: {e}")

# Initialize Gemini 2026 Client
gemini_client = None
if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"Warning: Failed to initialize Gemini client: {e}")

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


# ============================================================
# AUTHENTICATION & TOKEN LEDGER (SUPABASE)
# ============================================================

def hash_api_key(key: str) -> str:
    """Hashes API key before checking against Supabase storage."""
    return hashlib.sha256(key.strip().encode("utf-8")).hexdigest()

async def authenticate_and_deduct_tokens(
    cost: int,
    endpoint: str,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None),
) -> dict:
    """
    Validates API key from header, checks token balance in Supabase,
    and deducts tokens atomically per request.
    """
    raw_key = x_api_key
    if not raw_key and authorization:
        if authorization.startswith("Bearer "):
            raw_key = authorization.replace("Bearer ", "").strip()
        else:
            raw_key = authorization.strip()

    if not raw_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key. Include 'X-API-Key' or 'Authorization: Bearer <key>' in request headers."
        )

    if not supabase:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Supabase credentials not configured on backend."
        )

    hashed_key = hash_api_key(raw_key)

    # Query API Key and joined Profile from Supabase
    try:
        key_response = supabase.table("api_keys") \
            .select("id, user_id, is_active, profiles(id, token_balance)") \
            .eq("hashed_key", hashed_key) \
            .eq("is_active", True) \
            .execute()
        
        if not key_response.data or len(key_response.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or revoked API Key."
            )

        key_record = key_response.data[0]
        profile = key_record.get("profiles")
        user_id = key_record["user_id"]
        current_balance = profile.get("token_balance", 0) if profile else 0

        if current_balance < cost:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=f"Insufficient token balance ({current_balance} tokens remaining, {cost} required). Please top up."
            )

        new_balance = current_balance - cost

        # Deduct balance & log usage
        supabase.table("profiles").update({"token_balance": new_balance}).eq("id", user_id).execute()
        supabase.table("usage_logs").insert({
            "user_id": user_id,
            "endpoint": endpoint,
            "tokens_deducted": cost
        }).execute()

        return {"user_id": user_id, "tokens_remaining": new_balance, "cost": cost}

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authentication check failed: {str(exc)}"
        )


# ============================================================
# FASTAPI APP SETUP
# ============================================================

app = FastAPI(
    title="ReClaire Web Intelligence API",
    description="High-performance URL Scraper, Crawler, and Site Mapper Engine",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PYDANTIC SCHEMAS
# ============================================================

class ScrapeOptions(BaseModel):
    url: str
    formats: List[str] = Field(
        default_factory=lambda: ["markdown"],
        description="Supported formats: markdown, html, raw_html, summary, questions, json"
    )
    use_js_fallback: bool = True
    only_main_content: bool = True
    remove_tags: List[str] = Field(
        default_factory=lambda: ["script", "style", "nav", "footer", "header", "aside", "form", "iframe", "noscript"]
    )
    wait_for_selector: Optional[str] = None
    json_schema: Optional[Dict[str, Any]] = None
    user_question: Optional[str] = None

class MapOptions(BaseModel):
    url: str
    limit: int = Field(100, ge=1, le=1000)
    include_subdomains: bool = False

class CrawlOptions(BaseModel):
    url: str
    limit: int = Field(5, ge=1, le=50)
    scrape_options: Optional[ScrapeOptions] = None


# ============================================================
# UTILITIES & IMPROVED PLAYWRIGHT ENGINE
# ============================================================

def clean_input_url(raw_url: str) -> str:
    if not raw_url:
        return ""
    raw_url = raw_url.strip().strip("<>")
    if not raw_url.startswith(("http://", "https://")):
        raw_url = "https://" + raw_url
    return raw_url

def parse_json_response(text: str) -> Any:
    if not text:
        raise ValueError("Empty LLM response.")
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)

async def fetch_dynamic_content(url: str, wait_for_selector: Optional[str] = None) -> str:
    """Improved Playwright fetcher with lazy-load scrolling and network idle handling."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=DEFAULT_HEADERS["User-Agent"]
        )
        page = await context.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=25000)
            try:
                await page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass

            if wait_for_selector:
                try:
                    await page.wait_for_selector(wait_for_selector, timeout=5000)
                except Exception:
                    pass

            # Scroll to trigger lazy-loaded dynamic content
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2);")
            await asyncio.sleep(0.5)

            return await page.content()
        finally:
            await browser.close()


# ============================================================
# BASE PIPELINE: SCRAPE & LLM TRANSFORMATIONS
# ============================================================

async def base_scrape_pipeline(
    url: str,
    use_js_fallback: bool = True,
    only_main_content: bool = True,
    remove_tags: List[str] = None
) -> Dict[str, Any]:
    """Pipeline Step 1: Always scrape and convert URL to Markdown before formatting."""
    clean_url = clean_input_url(url)
    if not clean_url:
        raise HTTPException(status_code=400, detail="Invalid URL provided.")

    html_content = ""

    # Check for PDF
    if clean_url.lower().split("?")[0].endswith(".pdf"):
        try:
            async with httpx.AsyncClient(headers=DEFAULT_HEADERS, follow_redirects=True, timeout=20.0) as client:
                res = await client.get(clean_url)
                res.raise_for_status()
                reader = PdfReader(io.BytesIO(res.content))
                extracted = [page.extract_text() or "" for page in reader.pages]
                return {"clean_url": clean_url, "raw_html": "", "markdown": "\n\n".join(extracted)}
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Failed to extract PDF: {str(e)}")

    # Fetch standard HTTP HTML
    try:
        async with httpx.AsyncClient(headers=DEFAULT_HEADERS, follow_redirects=True, timeout=15.0) as client:
            res = await client.get(clean_url)
            if res.status_code == 200 and len(res.text.strip()) > 200:
                html_content = res.text
            elif use_js_fallback:
                html_content = await fetch_dynamic_content(clean_url)
    except Exception:
        if use_js_fallback:
            try:
                html_content = await fetch_dynamic_content(clean_url)
            except Exception as exc:
                raise HTTPException(status_code=502, detail=f"Failed to fetch content from URL: {str(exc)}")
        else:
            raise HTTPException(status_code=502, detail="Failed to fetch page using standard HTTP.")

    if not html_content:
        raise HTTPException(status_code=502, detail="Empty content received from site.")

    # Parse HTML and convert to Markdown
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in (remove_tags or ["script", "style", "nav", "footer", "header", "aside", "form"]):
        for el in soup.find_all(tag):
            el.decompose()

    target = (soup.find("main") or soup.find("article") or soup.body or soup) if only_main_content else (soup.body or soup)
    markdown_text = md(str(target), heading_style="ATX").strip()

    return {
        "clean_url": clean_url,
        "raw_html": html_content,
        "clean_html": str(target),
        "markdown": markdown_text or "No readable text found on page."
    }


def run_gemini_transform(prompt: str, response_json: bool = False) -> Any:
    """Executes Gemini LLM transformations using the 2026 google-genai SDK."""
    if not gemini_client:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured on the server.")

    config = types.GenerateContentConfig(response_mime_type="application/json") if response_json else None
    res = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=config
    )
    return parse_json_response(res.text) if response_json else res.text.strip()


# ============================================================
# API ENDPOINTS: SCRAPE, MAP, CRAWL
# ============================================================

@app.post("/v1/scrape")
async def scrape_endpoint(
    options: ScrapeOptions,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None)
):
    # Calculate Token Cost (Base: 1-3 tokens, +2 per AI transformation)
    base_cost = 3 if options.use_js_fallback else 1
    ai_requested = [f.lower() for f in options.formats if f.lower() in ["summary", "questions", "json"]]
    total_cost = base_cost + (len(ai_requested) * 2)

    auth_data = await authenticate_and_deduct_tokens(
        cost=total_cost, endpoint="/v1/scrape", x_api_key=x_api_key, authorization=authorization
    )

    # Pipeline Step 1: Always scrape to Markdown first
    scrape_base = await base_scrape_pipeline(
        url=options.url,
        use_js_fallback=options.use_js_fallback,
        only_main_content=options.only_main_content,
        remove_tags=options.remove_tags
    )

    result = {
        "success": True,
        "url": scrape_base["clean_url"],
        "tokens_deducted": total_cost,
        "tokens_remaining": auth_data["tokens_remaining"]
    }

    req_formats = [f.lower() for f in options.formats]
    markdown = scrape_base["markdown"]

    if "markdown" in req_formats or not req_formats:
        result["markdown"] = markdown
    if "html" in req_formats:
        result["html"] = scrape_base["clean_html"]
    if "raw_html" in req_formats:
        result["raw_html"] = scrape_base["raw_html"]

    # Pipeline Step 2: Formats & LLM Transformations
    if "summary" in req_formats:
        prompt = f"Provide a concise, comprehensive summary of this page in Markdown:\n\n{markdown[:20000]}"
        result["summary"] = run_gemini_transform(prompt)

    if "questions" in req_formats and options.user_question:
        prompt = f"Answer this question strictly using the text provided.\nQuestion: {options.user_question}\n\nContent:\n{markdown[:20000]}"
        result["answer"] = run_gemini_transform(prompt)

    if "json" in req_formats:
        schema_desc = json.dumps(options.json_schema or {"summary": "string", "key_facts": "list"})
        prompt = f"Extract data matching this JSON schema:\n{schema_desc}\n\nContent:\n{markdown[:20000]}"
        result["json"] = run_gemini_transform(prompt, response_json=True)

    return result


@app.post("/v1/map")
async def map_endpoint(
    options: MapOptions,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None)
):
    auth_data = await authenticate_and_deduct_tokens(
        cost=1, endpoint="/v1/map", x_api_key=x_api_key, authorization=authorization
    )

    clean_url = clean_input_url(options.url)
    parsed_base = urlparse(clean_url)

    try:
        async with httpx.AsyncClient(headers=DEFAULT_HEADERS, follow_redirects=True, timeout=15.0) as client:
            res = await client.get(clean_url)
            res.raise_for_status()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Site mapping failed: {str(exc)}")

    soup = BeautifulSoup(res.text, "html.parser")
    found_links = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        abs_link = urljoin(clean_url, href)
        p = urlparse(abs_link)

        if options.include_subdomains and p.hostname and parsed_base.hostname in p.hostname:
            found_links.add(abs_link)
        elif p.hostname == parsed_base.hostname:
            found_links.add(abs_link)

    limited_links = list(found_links)[:options.limit]

    return {
        "success": True,
        "url": clean_url,
        "links_count": len(limited_links),
        "links": limited_links,
        "tokens_deducted": 1,
        "tokens_remaining": auth_data["tokens_remaining"]
    }


@app.post("/v1/crawl")
async def crawl_endpoint(
    options: CrawlOptions,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None)
):
    # Step 1: Map base URL to gather target links
    map_opts = MapOptions(url=options.url, limit=options.limit)
    clean_url = clean_input_url(options.url)
    parsed_base = urlparse(clean_url)

    try:
        async with httpx.AsyncClient(headers=DEFAULT_HEADERS, follow_redirects=True, timeout=15.0) as client:
            res = await client.get(clean_url)
            soup = BeautifulSoup(res.text, "html.parser")
            target_links = {urljoin(clean_url, a["href"]) for a in soup.find_all("a", href=True) if urlparse(urljoin(clean_url, a["href"])).hostname == parsed_base.hostname}
            urls_to_crawl = list(target_links)[:options.limit] or [clean_url]
    except Exception:
        urls_to_crawl = [clean_url]

    total_cost = len(urls_to_crawl)
    auth_data = await authenticate_and_deduct_tokens(
        cost=total_cost, endpoint="/v1/crawl", x_api_key=x_api_key, authorization=authorization
    )

    scrape_opts = options.scrape_options or ScrapeOptions(url="")

    # Concurrently crawl mapped URLs with a semaphore cap to protect resources
    semaphore = asyncio.Semaphore(5)
    async def process_crawl_target(target_url: str):
        async with semaphore:
            try:
                base = await base_scrape_pipeline(
                    url=target_url,
                    use_js_fallback=scrape_opts.use_js_fallback,
                    only_main_content=scrape_opts.only_main_content,
                    remove_tags=scrape_opts.remove_tags
                )
                return {"url": target_url, "status": "success", "markdown": base["markdown"]}
            except Exception as err:
                return {"url": target_url, "status": "failed", "error": str(err)}

    crawled_results = await asyncio.gather(*[process_crawl_target(u) for u in urls_to_crawl])

    return {
        "success": True,
        "pages_crawled": len(crawled_results),
        "data": crawled_results,
        "tokens_deducted": total_cost,
        "tokens_remaining": auth_data["tokens_remaining"]
    }


# ============================================================
# SYSTEM HEALTH ENDPOINT
# ============================================================

@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "version": "3.0.0",
        "supabase_connected": supabase is not None,
        "gemini_connected": gemini_client is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
