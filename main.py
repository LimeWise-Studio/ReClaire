import asyncio
import io
import os
import json
import re
import hashlib
import secrets
import uuid
import zipfile
from datetime import datetime, timezone
import traceback
from typing import List, Optional, Dict, Any, Tuple
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from pypdf import PdfReader
from playwright.async_api import async_playwright

from fastapi import FastAPI, HTTPException, Header, status
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
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")
DEFAULT_TOKEN_BALANCE = int(os.environ.get("DEFAULT_TOKEN_BALANCE", "100"))
TEST_GRANT_TOKENS = int(os.environ.get("TEST_GRANT_TOKENS", "50"))
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  # <--- Replace or set ENV

# Initialize Supabase Client
supabase: Client = None
supabase_auth: Client = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        if SUPABASE_ANON_KEY:
            supabase_auth = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
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
# FASTAPI APP SETUP
# ============================================================

app = FastAPI(
    title="ReClaire Web Intelligence API",
    description="High-performance URL Scraper, Crawler, and Site Mapper Engine",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
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


class AuthCredentials(BaseModel):
    email: str
    password: str = Field(min_length=8, max_length=128)
    username: Optional[str] = Field(None, min_length=3, max_length=50)


class AuthResponse(BaseModel):
    success: bool
    user: Dict[str, Any]
    api_key: str
    tokens_remaining: int


class BatchScrapeOptions(BaseModel):
    urls: List[str] = Field(min_length=1, max_length=10)
    formats: List[str] = Field(default_factory=lambda: ["markdown"])
    use_js_fallback: bool = True
    only_main_content: bool = True
    remove_tags: List[str] = Field(
        default_factory=lambda: ["script", "style", "nav", "footer", "header", "aside", "form", "iframe", "noscript"]
    )
    wait_for_selector: Optional[str] = None
    json_schema: Optional[Dict[str, Any]] = None

# ============================================================
# AUTHENTICATION, TOKEN LEDGER & ACCOUNT API
# ============================================================

def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.strip().encode("utf-8")).hexdigest()


def _require_supabase() -> None:
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase credentials are not configured on the backend.")


def _extract_raw_credential(x_api_key: Optional[str], authorization: Optional[str]) -> Optional[str]:
    raw_key = x_api_key
    if not raw_key and authorization:
        raw = authorization.strip()
        raw_key = raw[7:].strip() if raw.lower().startswith("bearer ") else raw
    return raw_key


async def authenticate_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None),
) -> dict:
    _require_supabase()
    raw_key = _extract_raw_credential(x_api_key, authorization)
    if not raw_key:
        raise HTTPException(status_code=401, detail="Missing API key.")

    try:
        response = (
            supabase.table("api_keys")
            .select("id, user_id, is_active")
            .eq("hashed_key", hash_api_key(raw_key))
            .eq("is_active", True)
            .limit(1)
            .execute()
        )
        if not response.data:
            raise HTTPException(status_code=401, detail="Invalid or revoked API key.")

        record = response.data[0]
        profile_response = (
            supabase.table("profiles")
            .select("id, token_balance")
            .eq("id", record["user_id"])
            .limit(1)
            .execute()
        )
        if not profile_response.data:
            raise HTTPException(status_code=500, detail="User profile was not found.")

        return {
            "user_id": record["user_id"],
            "api_key_id": record["id"],
            "tokens_remaining": int(profile_response.data[0].get("token_balance") or 0),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Authentication failed: {exc}")


async def authenticate_and_deduct_tokens(
    cost: int,
    endpoint: str,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None),
) -> dict:
    if cost < 0:
        raise HTTPException(status_code=500, detail="Token cost cannot be negative.")

    auth = await authenticate_api_key(x_api_key, authorization)
    current_balance = auth["tokens_remaining"]
    user_id = auth["user_id"]

    if current_balance < cost:
        raise HTTPException(
            status_code=402,
            detail=f"Insufficient token balance ({current_balance} remaining, {cost} required).",
        )

    try:77777
        if cost:
            updated = (
                supabase.table("profiles")
                .update({"token_balance": current_balance - cost})
                .eq("id", user_id)
                .eq("token_balance", current_balance)
                .execute()
            )
            if not updated.data:
                raise HTTPException(
                    status_code=409,
                    detail="Token balance changed during the request. Please retry.",
                )
            new_balance = current_balance - cost
        else:
            new_balance = current_balance

        supabase.table("usage_logs").insert({
            "user_id": user_id,
            "endpoint": endpoint,
            "tokens_deducted": cost,
        }).execute()

        return {"user_id": user_id, "tokens_remaining": new_balance, "cost": cost}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Token accounting failed: {exc}")


def token_cost_for_scrape(formats: List[str], used_dynamic_fetch: bool = False) -> int:
    """ReClaire pricing: base scrape is 1 token, 3 tokens for Playwright JS fallback."""
    normalized = {f.strip().lower() for f in formats if f and f.strip()}
    cost = 3 if used_dynamic_fetch else 1
    
    if "html" in normalized: cost += 1
    if "raw_html" in normalized: cost += 1
    if "summary" in normalized: cost += 3
    if "questions" in normalized: cost += 3
    if "json" in normalized: cost += 2
    
    return cost


def normalize_formats(formats: List[str]) -> List[str]:
    aliases = {
        "md": "markdown",
        "q&a": "questions",
        "qa": "questions",
        "q_a": "questions",
        "structured_json": "json",
    }
    return [aliases.get(f.strip().lower(), f.strip().lower()) for f in formats if f and f.strip()]

async def commit_token_deduction(user_id: str, current_balance: int, cost: int, endpoint: str) -> int:
    """Phase 3: Deduct tokens strictly after successful execution."""
    if cost <= 0:
        return current_balance
    new_balance = current_balance - cost
    updated = (
        supabase.table("profiles")
        .update({"token_balance": new_balance})
        .eq("id", user_id)
        .eq("token_balance", current_balance)
        .execute()
    )
    if not updated.data:
        raise HTTPException(status_code=409, detail="Token balance mismatch during deduction.")
    
    supabase.table("usage_logs").insert({
        "user_id": user_id, "endpoint": endpoint, "tokens_deducted": cost
    }).execute()
    
    return new_balance


def generate_api_key() -> Tuple[str, str]:
    raw = "rc_" + secrets.token_urlsafe(32)
    return raw, hash_api_key(raw)


def profile_for_user(user_id: str, username: Optional[str] = None) -> Dict[str, Any]:
    response = supabase.table("profiles").select("*").eq("id", user_id).limit(1).execute()
    if response.data:
        return response.data[0]

    # Initialize new user with 100 tokens and 0 purchase points
    new_profile = {
        "id": user_id,
        "token_balance": DEFAULT_TOKEN_BALANCE,
        "purchase_points": 0
    }
    if username:
        new_profile["username"] = username

    created = supabase.table("profiles").insert(new_profile).execute()
    if not created.data:
        raise HTTPException(status_code=500, detail="Could not create user profile.")
    return created.data[0]


async def issue_api_key(user_id: str) -> str:
    raw_key, hashed_key = generate_api_key()
    supabase.table("api_keys").update({"is_active": False}).eq("user_id", user_id).execute()
    inserted = supabase.table("api_keys").insert({
        "user_id": user_id,
        "hashed_key": hashed_key,
        "is_active": True,
    }).execute()
    if not inserted.data:
        raise HTTPException(status_code=500, detail="Could not create API key.")
    return raw_key


@app.post("/auth/register", response_model=AuthResponse)
async def register(credentials: AuthCredentials):
    _require_supabase()
    if not supabase_auth:
        raise HTTPException(status_code=500, detail="SUPABASE_ANON_KEY is required for authentication.")

    email = credentials.email.strip().lower()
    try:
        created = supabase.auth.admin.create_user({
            "email": email,
            "password": credentials.password,
            "email_confirm": True,
        })
        user = created.user
        if not user:
            raise HTTPException(status_code=500, detail="Supabase did not return a user.")

        profile = profile_for_user(user.id, credentials.username)
        api_key = await issue_api_key(user.id)
        return {
            "success": True,
            "user": {"id": user.id, "email": user.email},
            "api_key": api_key,
            "tokens_remaining": int(profile.get("token_balance") or 0),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Registration failed: {exc}")


@app.post("/auth/login", response_model=AuthResponse)
async def login(credentials: AuthCredentials):
    _require_supabase()
    if not SUPABASE_ANON_KEY:
        raise HTTPException(status_code=500, detail="SUPABASE_ANON_KEY is required for authentication.")

    try:
        auth_client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        session = auth_client.auth.sign_in_with_password({
            "email": credentials.email.strip().lower(),
            "password": credentials.password,
        })
        user = session.user
        if not user:
            raise HTTPException(status_code=401, detail="Invalid email or password.")

        profile = profile_for_user(user.id)
        api_key = await issue_api_key(user.id)
        return {
            "success": True,
            "user": {"id": user.id, "email": user.email},
            "api_key": api_key,
            "tokens_remaining": int(profile.get("token_balance") or 0),
        }
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid email or password.")


@app.post("/auth/logout")
async def logout(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None),
):
    auth = await authenticate_api_key(x_api_key, authorization)
    supabase.table("api_keys").update({"is_active": False}).eq("id", auth["api_key_id"]).execute()
    return {"success": True}


@app.get("/auth/me")
async def me(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None),
):
    auth = await authenticate_api_key(x_api_key, authorization)
    return {
        "success": True,
        "user_id": auth["user_id"],
        "tokens_remaining": auth["tokens_remaining"],
    }


@app.get("/v1/usage")
async def usage(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None),
):
    auth = await authenticate_api_key(x_api_key, authorization)
    logs = (
        supabase.table("usage_logs")
        .select("id, endpoint, tokens_deducted, created_at")
        .eq("user_id", auth["user_id"])
        .order("created_at", desc=True)
        .limit(100)
        .execute()
    )
    return {
        "success": True,
        "tokens_remaining": auth["tokens_remaining"],
        "logs": logs.data or [],
    }


@app.post("/v1/testing/grant")
async def grant_testing_tokens(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None),
):
    auth = await authenticate_api_key(x_api_key, authorization)
    try:
        marker = (
            supabase.table("usage_logs")
            .select("id")
            .eq("user_id", auth["user_id"])
            .eq("endpoint", "/v1/testing/grant")
            .limit(1)
            .execute()
        )
        if marker.data:
            raise HTTPException(status_code=409, detail="Testing token grant has already been claimed.")

        new_balance = auth["tokens_remaining"] + TEST_GRANT_TOKENS
        updated = (
            supabase.table("profiles")
            .update({"token_balance": new_balance})
            .eq("id", auth["user_id"])
            .eq("token_balance", auth["tokens_remaining"])
            .execute()
        )
        if not updated.data:
            raise HTTPException(status_code=409, detail="Balance changed. Please retry.")

        # A zero-cost ledger entry acts as the one-time claim marker.
        supabase.table("usage_logs").insert({
            "user_id": auth["user_id"],
            "endpoint": "/v1/testing/grant",
            "tokens_deducted": 0,
        }).execute()

        return {
            "success": True,
            "tokens_granted": TEST_GRANT_TOKENS,
            "tokens_remaining": new_balance,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Testing grant failed: {exc}")


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
    used_dynamic_fetch = False

    # Check for PDF
    if clean_url.lower().split("?")[0].endswith(".pdf"):
        try:
            async with httpx.AsyncClient(headers=DEFAULT_HEADERS, follow_redirects=True, timeout=20.0) as client:
                res = await client.get(clean_url)
                res.raise_for_status()
                reader = PdfReader(io.BytesIO(res.content))
                extracted = [page.extract_text() or "" for page in reader.pages]
                return {"clean_url": clean_url, "raw_html": "", "markdown": "\n\n".join(extracted), "used_dynamic_fetch": False}
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
                used_dynamic_fetch = True
    except Exception:
        if use_js_fallback:
            try:
                html_content = await fetch_dynamic_content(clean_url)
                used_dynamic_fetch = True
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
        "markdown": markdown_text or "No readable text found on page.",
        "used_dynamic_fetch": used_dynamic_fetch
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
    options.formats = normalize_formats(options.formats)
    allowed_formats = {"markdown", "html", "raw_html", "summary", "questions", "json"}
    if unknown := [f for f in options.formats if f not in allowed_formats]:
        raise HTTPException(status_code=422, detail=f"Unsupported format(s): {', '.join(unknown)}")
    
    # PHASE 1: Authenticate & Pre-check Balance
    auth = await authenticate_api_key(x_api_key, authorization)
    user_id = auth["user_id"]
    current_balance = auth["tokens_remaining"]
    
    # Calculate the maximum possible cost (assuming JS fallback triggers) to prevent negative balances
    max_possible_cost = token_cost_for_scrape(options.formats, used_dynamic_fetch=True)
    if current_balance < max_possible_cost:
        raise HTTPException(status_code=402, detail=f"Insufficient tokens. Ensure you have at least {max_possible_cost} tokens.")

    # (Lightweight Priority Logic) - In the future, check profile['purchase_points'] here
    # High-point users can bypass an asyncio.sleep() delay or get assigned to a premium Playwright semaphore pool.

    # PHASE 2: Execute Scrape & LLM Transformations
    scrape_base = await base_scrape_pipeline(
        url=options.url, use_js_fallback=options.use_js_fallback,
        only_main_content=options.only_main_content, remove_tags=options.remove_tags
    )
    
    req_formats = [f.lower() for f in options.formats]
    markdown = scrape_base["markdown"]
    result_data = {"success": True, "url": scrape_base["clean_url"]}

    if "markdown" in req_formats or not req_formats: result_data["markdown"] = markdown
    if "html" in req_formats: result_data["html"] = scrape_base["clean_html"]
    if "raw_html" in req_formats: result_data["raw_html"] = scrape_base["raw_html"]
    
    if "summary" in req_formats:
        result_data["summary"] = run_gemini_transform(f"Provide a summary:\n\n{markdown[:20000]}")
    if "questions" in req_formats and options.user_question:
        result_data["answer"] = run_gemini_transform(f"Answer: {options.user_question}\n\nContent:\n{markdown[:20000]}")
    if "json" in req_formats:
        result_data["json"] = run_gemini_transform(f"Extract JSON:\n{json.dumps(options.json_schema)}\n\nContent:\n{markdown[:20000]}", response_json=True)

    # PHASE 3: Deduct & Commit Tokens (Only happens if Phase 2 fully succeeds)
    actual_cost = token_cost_for_scrape(options.formats, scrape_base["used_dynamic_fetch"])
    new_balance = await commit_token_deduction(user_id, current_balance, actual_cost, "/v1/scrape")

    result_data["tokens_deducted"] = actual_cost
    result_data["tokens_remaining"] = new_balance
    
    return result_data


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

    # Crawling is billed per page, so authenticate up front and defer the
    # deduction until we know how many pages were successfully crawled.
    await authenticate_api_key(x_api_key, authorization)

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

    # Site Crawling pricing: 1 token per discovered/crawled page.
    # Total cost = 1 x N, where N is the number of pages successfully crawled.
    successful_pages = sum(1 for r in crawled_results if r["status"] == "success")
    total_cost = 1 * successful_pages

    auth_data = await authenticate_and_deduct_tokens(
        cost=total_cost, endpoint="/v1/crawl", x_api_key=x_api_key, authorization=authorization
    )

    return {
        "success": True,
        "pages_crawled": len(crawled_results),
        "data": crawled_results,
        "tokens_deducted": total_cost,
        "tokens_remaining": auth_data["tokens_remaining"]
    }



# ============================================================
# BATCH SCRAPING
# ============================================================

# Temporary store keeps batch Markdown independent of a new database table.
# For horizontally scaled production, move this store to Supabase/object storage.
BATCH_STORE: Dict[str, Dict[str, Any]] = {}


@app.post("/v1/batch-scrape")
async def batch_scrape_endpoint(
    options: BatchScrapeOptions,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None),
):
    formats = normalize_formats(options.formats)
    allowed_formats = {"markdown", "html", "raw_html", "summary", "questions", "json"}
    unknown = [f for f in formats if f not in allowed_formats]
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unsupported format(s): {', '.join(unknown)}")
    if "questions" in formats:
        raise HTTPException(status_code=422, detail="Q&A is not supported by batch scraping.")

    auth = await authenticate_api_key(x_api_key, authorization)
    batch_id = str(uuid.uuid4())

    BATCH_STORE[batch_id] = {
        "user_id": auth["user_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": [],
        "markdown": {},
    }

    # One row = one scrape operation and therefore one token base charge,
    # plus the selected format charges.
    for index, raw_url in enumerate(options.urls):
        row = {
            "index": index,
            "url": clean_input_url(raw_url),
            "status": "scraping",
            "error": None,
        }
        BATCH_STORE[batch_id]["rows"].append(row)

        if not row["url"]:
            row["status"] = "failed"
            row["error"] = "Invalid URL."
            continue

        try:
            # Scrape first, then charge only for a successful operation.
            base = await base_scrape_pipeline(
                url=row["url"],
                use_js_fallback=options.use_js_fallback,
                only_main_content=options.only_main_content,
                remove_tags=options.remove_tags,
            )
            cost = token_cost_for_scrape(formats, base["used_dynamic_fetch"])
            auth_data = await authenticate_and_deduct_tokens(
                cost=cost,
                endpoint="/v1/batch-scrape",
                x_api_key=x_api_key,
                authorization=authorization,
            )

            BATCH_STORE[batch_id]["markdown"][index] = base["markdown"]
            row["url"] = base["clean_url"]
            row["status"] = "complete"
            row["tokens_deducted"] = cost
            row["tokens_remaining"] = auth_data["tokens_remaining"]
        except HTTPException as exc:
            row["status"] = "failed"
            row["error"] = str(exc.detail)
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)

    final_auth = await authenticate_api_key(x_api_key, authorization)
    return {
        "success": True,
        "batch_id": batch_id,
        "total": len(options.urls),
        "completed": sum(r["status"] == "complete" for r in BATCH_STORE[batch_id]["rows"]),
        "failed": sum(r["status"] == "failed" for r in BATCH_STORE[batch_id]["rows"]),
        "rows": [
            {k: v for k, v in row.items() if k not in {"tokens_deducted", "tokens_remaining"}}
            for row in BATCH_STORE[batch_id]["rows"]
        ],
        "tokens_remaining": final_auth["tokens_remaining"],
    }


@app.get("/v1/batch-scrape/{batch_id}")
async def batch_status(
    batch_id: str,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None),
):
    auth = await authenticate_api_key(x_api_key, authorization)
    batch = BATCH_STORE.get(batch_id)
    if not batch or batch["user_id"] != auth["user_id"]:
        raise HTTPException(status_code=404, detail="Batch not found.")

    return {
        "success": True,
        "batch_id": batch_id,
        "rows": [
            {k: v for k, v in row.items() if k not in {"tokens_deducted", "tokens_remaining"}}
            for row in batch["rows"]
        ],
        "completed": sum(r["status"] == "complete" for r in batch["rows"]),
        "failed": sum(r["status"] == "failed" for r in batch["rows"]),
        "tokens_remaining": auth["tokens_remaining"],
    }


@app.get("/v1/batch-scrape/{batch_id}/download")
async def download_batch_markdown(
    batch_id: str,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None),
):
    auth = await authenticate_api_key(x_api_key, authorization)
    batch = BATCH_STORE.get(batch_id)
    if not batch or batch["user_id"] != auth["user_id"]:
        raise HTTPException(status_code=404, detail="Batch not found.")

    completed = [row for row in batch["rows"] if row["status"] == "complete"]
    if not completed:
        raise HTTPException(status_code=409, detail="No completed Markdown files are available.")

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for row in completed:
            index = row["index"]
            parsed = urlparse(row["url"])
            base_name = (parsed.netloc + parsed.path).strip("/") or f"page_{index + 1}"
            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", base_name)[:100]
            if not safe_name.lower().endswith(".md"):
                safe_name += ".md"
            zf.writestr(f"{index + 1:02d}_{safe_name}", batch["markdown"][index])

    archive.seek(0)
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        archive,
        media_type="application/zip",
        headers={
            "Content-Disposition":
                f'attachment; filename="reclaire-batch-{batch_id[:8]}.zip"'
        },
    )



# ============================================================
# SYSTEM HEALTH ENDPOINT
# ============================================================

@app.get("/v1/stats/public")
async def public_stats():
    """Fetches the total number of registered ReClaire users for the landing page hero."""
    try:
        response = supabase.table("profiles").select("id", count="exact").execute()
        return {"success": True, "total_users": response.count if response.count else 0}
    except Exception as e:
        return {"success": False, "total_users": 0, "error": "Could not fetch user stats."}

    
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
