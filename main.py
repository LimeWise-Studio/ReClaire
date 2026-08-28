import asyncio
import io
import os
import json
import base64
import re
import sqlite3
import hashlib
import secrets
import traceback
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin, urlparse
from typing import List, Optional, Dict, Any

import httpx
import jwt
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from pypdf import PdfReader
from playwright.async_api import async_playwright

from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ============================================================
# CONFIGURATION & CONSTANTS
# ============================================================

SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "reclaire-super-secret-jwt-key-2026")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days
DB_PATH = os.environ.get("RECLAIRE_DB_PATH", "reclaire.db")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


# ============================================================
# GEMINI SETUP
# ============================================================

try:
    from google import genai
    from google.genai import types

    gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
except ImportError:
    genai = None
    types = None
    gemini_client = None


# ============================================================
# DATABASE & TOKEN LEDGER SETUP (SQLite)
# ============================================================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Users Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        tokens INTEGER NOT NULL DEFAULT 100,
        created_at TEXT NOT NULL
    )
    """)
    
    # Token Audit Transactions
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS token_transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        amount INTEGER NOT NULL,
        action TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    """)
    
    conn.commit()
    conn.close()

init_db()


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# ============================================================
# AUTHENTICATION HELPERS
# ============================================================

security = HTTPBearer(auto_error=False)

def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
    return f"{salt}:{pwd_hash.hex()}"

def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, original_hash = stored_hash.split(":")
        pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
        return pwd_hash.hex() == original_hash
    except Exception:
        return False

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(
    auth: Optional[HTTPAuthorizationCredentials] = Security(security),
    conn: sqlite3.Connection = Depends(get_db)
) -> dict:
    if not auth:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide a Bearer token in the Authorization header.",
        )
    try:
        payload = jwt.decode(auth.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid auth token payload.")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired access token.")

    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email, tokens FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    
    if not user:
        raise HTTPException(status_code=401, detail="User account not found.")
        
    return dict(user)


def deduct_tokens(user_id: int, cost: int, action: str, conn: sqlite3.Connection):
    """Deduct tokens atomically or throw an HTTP 402 exception."""
    cursor = conn.cursor()
    cursor.execute("SELECT tokens FROM users WHERE id = ?", (user_id,))
    res = cursor.fetchone()
    
    if not res:
        raise HTTPException(status_code=404, detail="User not found.")
        
    current_tokens = res["tokens"]
    
    if current_tokens < cost:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=f"Insufficient tokens. Required: {cost}, Available: {current_tokens}. Top up your balance to proceed."
        )
        
    new_balance = current_tokens - cost
    now_iso = datetime.now(timezone.utc).isoformat()
    
    cursor.execute("UPDATE users SET tokens = ? WHERE id = ?", (new_balance, user_id))
    cursor.execute(
        "INSERT INTO token_transactions (user_id, amount, action, timestamp) VALUES (?, ?, ?, ?)",
        (user_id, -cost, action, now_iso)
    )
    conn.commit()
    return new_balance


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="ReClaire API",
    description="Micro-SaaS Web Scraping, AI Extraction, and Autonomous Agent Engine",
    version="2.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=FileResponse)
async def read_root():
    return FileResponse("signup.html")

@app.get("/signup", response_class=FileResponse)
async def read_signup():
    return FileResponse("signup.html")


# ============================================================
# PYDANTIC MODELS
# ============================================================

class UserRegister(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    username_or_email: str
    password: str

class TopupRequest(BaseModel):
    amount: int = Field(..., ge=10, le=10000, description="Tokens to add")

class ScrapeOptions(BaseModel):
    url: str
    formats: List[str] = Field(
        default_factory=lambda: ["markdown"],
        description="markdown, html, raw_html, links, images, screenshot, summary, highlights, json, questions, branding"
    )
    use_js_fallback: bool = True
    only_main_content: bool = True
    remove_tags: List[str] = Field(
        default_factory=lambda: ["script", "style", "nav", "footer", "header", "aside", "form", "iframe", "noscript"]
    )
    wait_for_selector: Optional[str] = None
    json_schema: Optional[Dict[str, Any]] = None
    user_question: Optional[str] = None
    web_augmented_qa: bool = True

class BatchScrapeRequest(BaseModel):
    urls: List[str]
    options: Optional[ScrapeOptions] = None

class MapOptions(BaseModel):
    url: str
    limit: int = Field(100, ge=1, le=500)
    include_subdomains: bool = False

class CrawlOptions(BaseModel):
    url: str
    limit: int = Field(5, ge=1, le=25)
    scrape_options: Optional[ScrapeOptions] = None

class SearchOptions(BaseModel):
    query: str
    limit: int = Field(5, ge=1, le=25)
    category: str = "web"

class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str

class AgentOptions(BaseModel):
    prompt: str
    urls: Optional[List[str]] = Field(None, description="Seed URLs for Clara")
    output_format: str = Field("chat", description="'chat', 'json', or 'csv'")
    json_schema: Optional[Dict[str, Any]] = None
    chat_history: Optional[List[ChatMessage]] = Field(default_factory=list, description="Previous dialogue turns")


# ============================================================
# GENERAL UTILITIES & EXTRACTION HELPERS
# ============================================================

def clean_input_url(raw_url: str) -> str:
    if not raw_url:
        return ""
    raw_url = raw_url.strip().strip("<>")
    if not raw_url:
        return ""
    if not raw_url.startswith(("http://", "https://")):
        raw_url = "https://" + raw_url
    return raw_url

def safe_text(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, list):
        value = " ".join(str(v) for v in value)
    else:
        value = str(value)
    return re.sub(r"\s+", " ", value).strip()

def unique_list(values: List[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result

def absolute_url(base_url: str, value: Optional[str]) -> Optional[str]:
    if not value or value.startswith(("data:", "blob:", "javascript:", "mailto:", "tel:")):
        return None
    return urljoin(base_url, value.strip())

def is_http_url(value: Optional[str]) -> bool:
    if not value:
        return False
    try:
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.hostname)
    except Exception:
        return False

def parse_json_response(text: str) -> Any:
    if not text:
        raise ValueError("Gemini returned an empty response.")
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)

def parse_pdf_bytes(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    extracted = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            extracted.append(f"## Page {idx + 1}\n\n{text.strip()}")
    return "\n\n".join(extracted)


# ============================================================
# PLAYWRIGHT ENGINE
# ============================================================

async def fetch_dynamic_content(url: str, wait_for_selector: Optional[str] = None, take_screenshot: bool = False) -> dict:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=DEFAULT_HEADERS["User-Agent"]
        )
        page = await context.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            try:
                await page.wait_for_load_state("networkidle", timeout=8000)
            except Exception:
                pass

            if wait_for_selector:
                try:
                    await page.wait_for_selector(wait_for_selector, timeout=5000)
                except Exception:
                    pass

            content = await page.content()
            screenshot_b64 = None
            if take_screenshot:
                try:
                    shot_bytes = await page.screenshot(full_page=True)
                    screenshot_b64 = base64.b64encode(shot_bytes).decode("utf-8")
                except Exception:
                    screenshot_b64 = None

            return {"html": content, "screenshot": screenshot_b64}
        finally:
            await browser.close()


# ============================================================
# MEDIA & BRANDING EXTRACTION
# ============================================================

def _srcset_urls(value: Optional[str], base_url: str) -> List[str]:
    if not value:
        return []
    urls = []
    for candidate in str(value).split(","):
        candidate = candidate.strip()
        if candidate:
            url = absolute_url(base_url, candidate.split()[0])
            if url and is_http_url(url):
                urls.append(url)
    return urls

def _image_candidates(img, base_url: str) -> List[str]:
    candidates = []
    for attr in ("src", "data-src", "data-lazy-src", "data-original", "data-image", "data-url"):
        val = img.get(attr)
        if val:
            url = absolute_url(base_url, val)
            if url and is_http_url(url):
                candidates.append(url)
    candidates.extend(_srcset_urls(img.get("srcset"), base_url))
    candidates.extend(_srcset_urls(img.get("data-srcset"), base_url))
    return unique_list(candidates)

def extract_image_metadata(soup: BeautifulSoup, base_url: str) -> List[Dict[str, Any]]:
    results = []
    for img in soup.find_all("img"):
        candidates = _image_candidates(img, base_url)
        if candidates:
            results.append({
                "url": candidates[0],
                "src_candidates": candidates[:6],
                "alt": safe_text(img.get("alt")),
                "title": safe_text(img.get("title")),
                "width": img.get("width"),
                "height": img.get("height"),
            })
    seen = set()
    cleaned = []
    for item in results:
        if item["url"] not in seen:
            seen.add(item["url"])
            cleaned.append(item)
    return cleaned

def extract_branding_assets(soup: BeautifulSoup, base_url: str, domain: str) -> Dict[str, Any]:
    assets = {
        "domain": domain, "site_name": None, "title": None, "description": None,
        "favicon": None, "apple_touch_icon": None, "logo": None, "logo_type": None,
        "logo_candidates": [], "hero_image": None, "og_image": None, "theme_color": None, "meta": {}
    }
    if soup.title:
        assets["title"] = safe_text(soup.title.get_text())

    def get_meta(name=None, property_name=None):
        tag = soup.find("meta", attrs={"name": name}) if name else soup.find("meta", attrs={"property": property_name})
        return safe_text(tag.get("content")) if tag else None

    assets["description"] = get_meta(name="description") or get_meta(property_name="og:description")
    assets["site_name"] = get_meta(property_name="og:site_name") or get_meta(name="application-name")
    assets["og_image"] = absolute_url(base_url, get_meta(property_name="og:image"))
    assets["hero_image"] = assets["og_image"]
    assets["theme_color"] = get_meta(name="theme-color")

    for link in soup.find_all("link"):
        rel = link.get("rel", [])
        rel_set = {str(r).lower() for r in (rel if isinstance(rel, list) else [rel])}
        href = link.get("href")
        if href and ("icon" in rel_set or "apple-touch-icon" in rel_set):
            icon_url = absolute_url(base_url, href)
            if icon_url and is_http_url(icon_url):
                if "apple-touch-icon" in rel_set:
                    assets["apple_touch_icon"] = icon_url
                if not assets["favicon"]:
                    assets["favicon"] = icon_url

    if not assets["favicon"]:
        assets["favicon"] = f"[https://www.google.com/s2/favicons?domain=](https://www.google.com/s2/favicons?domain=){domain}&sz=64"

    return assets


# ============================================================
# GEMINI SERVICE CALLS
# ============================================================

def require_gemini():
    if not gemini_client:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured on the server."
        )

def generate_gemini_qa(markdown_text: str, user_question: str, web_augmented: bool) -> str:
    require_gemini()
    prompt = f"""
Answer the user question based on this content:
Question: {user_question}

Content:
{markdown_text[:20000]}
"""
    res = gemini_client.models.generate_content(model="gemini-3.6-flash", contents=prompt)
    return res.text.strip() if res.text else ""

def generate_gemini_branding(markdown_text: str, branding_assets: Dict[str, Any]) -> Dict[str, Any]:
    require_gemini()
    prompt = f"""
Create a brand profile from these assets:
{json.dumps(branding_assets)[:10000]}

Page Markdown:
{markdown_text[:10000]}
"""
    res = gemini_client.models.generate_content(
        model="gemini-3.6-flash", contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    return parse_json_response(res.text)

def generate_gemini_summary(markdown_text: str) -> str:
    require_gemini()
    prompt = f"Provide a concise summary of this webpage:\n\n{markdown_text[:20000]}"
    res = gemini_client.models.generate_content(model="gemini-3.6-flash", contents=prompt)
    return res.text.strip() if res.text else ""

def generate_gemini_highlights(markdown_text: str) -> Any:
    require_gemini()
    prompt = f"Extract 3-5 key highlights as a JSON list of strings from:\n\n{markdown_text[:15000]}"
    res = gemini_client.models.generate_content(
        model="gemini-3.6-flash", contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    return parse_json_response(res.text)

def generate_gemini_json_schema(markdown_text: str, json_schema: Dict[str, Any]) -> dict:
    require_gemini()
    prompt = f"Extract JSON according to this schema:\n{json.dumps(json_schema)}\n\nContent:\n{markdown_text[:15000]}"
    res = gemini_client.models.generate_content(
        model="gemini-3.6-flash", contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    return parse_json_response(res.text)


# ============================================================
# CORE SCRAPING ENGINE
# ============================================================

async def scrape_engine(
    url: str,
    use_js_fallback: bool = True,
    only_main_content: bool = True,
    remove_tags: List[str] = None,
    wait_for_selector: Optional[str] = None,
    json_schema: Optional[Dict[str, Any]] = None,
    formats: Optional[List[str]] = None,
    user_question: Optional[str] = None,
    web_augmented_qa: bool = True,
) -> dict:
    if remove_tags is None:
        remove_tags = ["script", "style", "nav", "footer", "header", "aside", "form", "iframe", "noscript"]

    req_formats = list(dict.fromkeys([str(f).lower().strip() for f in (formats or ["markdown"])]))
    clean_url = clean_input_url(url)
    if not clean_url:
        raise HTTPException(status_code=400, detail="A valid URL is required.")

    parsed = urlparse(clean_url)
    parsed_domain = parsed.hostname or "unknown"
    result = {"success": True, "url": clean_url}
    
    screenshot_b64, clean_markdown, html_content = None, "", ""
    meta_info = {"domain": parsed_domain, "favicon": f"[https://www.google.com/s2/favicons?domain=](https://www.google.com/s2/favicons?domain=){parsed_domain}&sz=64"}

    # Handle PDF
    if clean_url.lower().split("?")[0].endswith(".pdf"):
        try:
            async with httpx.AsyncClient(headers=DEFAULT_HEADERS, follow_redirects=True, timeout=30.0) as client:
                res = await client.get(clean_url)
                res.raise_for_status()
                clean_markdown = parse_pdf_bytes(res.content)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Failed to fetch PDF: {exc}")
    else:
        # Handle Webpage
        try:
            async with httpx.AsyncClient(headers=DEFAULT_HEADERS, follow_redirects=True, timeout=20.0) as client:
                res = await client.get(clean_url)
                if res.status_code == 200:
                    html_content = res.text
                elif use_js_fallback:
                    pw = await fetch_dynamic_content(clean_url, wait_for_selector, "screenshot" in req_formats)
                    html_content, screenshot_b64 = pw["html"], pw["screenshot"]
        except Exception:
            if use_js_fallback:
                try:
                    pw = await fetch_dynamic_content(clean_url, wait_for_selector, "screenshot" in req_formats)
                    html_content, screenshot_b64 = pw["html"], pw["screenshot"]
                except Exception as exc:
                    raise HTTPException(status_code=502, detail=f"Web fetch failed: {exc}")

        if not html_content:
            raise HTTPException(status_code=502, detail="Unable to retrieve HTML content.")

        soup_raw = BeautifulSoup(html_content, "html.parser")
        branding_assets = extract_branding_assets(soup_raw, clean_url, parsed_domain)
        meta_info.update(branding_assets)
        result["title"] = branding_assets.get("title") or parsed_domain

        if "links" in req_formats:
            links = [absolute_url(clean_url, a.get("href")) for a in soup_raw.find_all("a", href=True)]
            result["links"] = unique_list([l for l in links if l])

        image_meta = extract_image_metadata(soup_raw, clean_url)
        if "images" in req_formats or "branding" in req_formats:
            result["images"] = image_meta
            meta_info["extracted_images"] = image_meta[:10]

        soup_content = BeautifulSoup(str(soup_raw), "html.parser")
        for tag in set(["script", "style", "noscript", "iframe"] + remove_tags):
            for el in soup_content.find_all(tag):
                try:
                    el.decompose()
                except Exception:
                    pass

        target = (soup_content.find("main") or soup_content.find("article") or soup_content.body or soup_content) if only_main_content else (soup_content.body or soup_content)
        clean_markdown = md(str(target), heading_style="ATX").strip() or "No readable content extracted."

        if "html" in req_formats:
            result["html"] = str(target)
        if "raw_html" in req_formats:
            result["raw_html"] = html_content
        if "screenshot" in req_formats and screenshot_b64:
            result["screenshot"] = f"data:image/png;base64,{screenshot_b64}"

    result["metadata"] = meta_info
    if "markdown" in req_formats or not req_formats:
        result["markdown"] = clean_markdown

    # LLM Transformations
    if clean_markdown:
        if "questions" in req_formats and user_question:
            try:
                result["qa_answer"] = await asyncio.to_thread(generate_gemini_qa, clean_markdown, user_question, web_augmented_qa)
            except Exception as e:
                result["qa_error"] = str(e)
        if "summary" in req_formats:
            try:
                result["summary"] = await asyncio.to_thread(generate_gemini_summary, clean_markdown)
            except Exception as e:
                result["summary_error"] = str(e)
        if "highlights" in req_formats:
            try:
                result["highlights"] = await asyncio.to_thread(generate_gemini_highlights, clean_markdown)
            except Exception as e:
                result["highlights_error"] = str(e)
        if "json" in req_formats:
            try:
                result["json_data"] = await asyncio.to_thread(generate_gemini_json_schema, clean_markdown, json_schema or {"title": "str"})
            except Exception as e:
                result["json_error"] = str(e)
        if "branding" in req_formats:
            try:
                result["branding"] = {
                    "visual_assets": meta_info,
                    "brand_analysis": await asyncio.to_thread(generate_gemini_branding, clean_markdown, meta_info)
                }
            except Exception as e:
                result["branding_error"] = str(e)

    result["requested_formats"] = req_formats
    return result


# ============================================================
# AUTHENTICATION ENDPOINTS
# ============================================================

@app.post("/auth/register")
async def register(user: UserRegister, conn: sqlite3.Connection = Depends(get_db)):
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", (user.username, user.email))
    if cursor.fetchone():
        raise HTTPException(status_code=400, detail="Username or email is already registered.")

    pwd_hash = hash_password(user.password)
    now_iso = datetime.now(timezone.utc).isoformat()
    initial_tokens = 100

    cursor.execute(
        "INSERT INTO users (username, email, password_hash, tokens, created_at) VALUES (?, ?, ?, ?, ?)",
        (user.username, user.email, pwd_hash, initial_tokens, now_iso)
    )
    user_id = cursor.lastrowid
    cursor.execute(
        "INSERT INTO token_transactions (user_id, amount, action, timestamp) VALUES (?, ?, ?, ?)",
        (user_id, initial_tokens, "Registration Bonus", now_iso)
    )
    conn.commit()

    token = create_access_token({"sub": user_id, "username": user.username})
    return {
        "success": True,
        "message": "Account created successfully.",
        "access_token": token,
        "token_type": "bearer",
        "user": {"id": user_id, "username": user.username, "tokens": initial_tokens}
    }


@app.post("/auth/login")
async def login(user_data: UserLogin, conn: sqlite3.Connection = Depends(get_db)):
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, username, password_hash, tokens FROM users WHERE username = ? OR email = ?",
        (user_data.username_or_email, user_data.username_or_email)
    )
    user = cursor.fetchone()

    if not user or not verify_password(user_data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username/email or password.")

    token = create_access_token({"sub": user["id"], "username": user["username"]})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {"id": user["id"], "username": user["username"], "tokens": user["tokens"]}
    }


@app.get("/auth/me")
async def get_me(
    current_user: dict = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db)
):
    cursor = conn.cursor()
    cursor.execute("SELECT amount, action, timestamp FROM token_transactions WHERE user_id = ? ORDER BY id DESC LIMIT 10", (current_user["id"],))
    txs = [dict(r) for r in cursor.fetchall()]
    
    return {
        "user": current_user,
        "recent_transactions": txs
    }


@app.post("/auth/topup")
async def topup_tokens(
    req: TopupRequest,
    current_user: dict = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db)
):
    cursor = conn.cursor()
    new_balance = current_user["tokens"] + req.amount
    now_iso = datetime.now(timezone.utc).isoformat()
    
    cursor.execute("UPDATE users SET tokens = ? WHERE id = ?", (new_balance, current_user["id"]))
    cursor.execute(
        "INSERT INTO token_transactions (user_id, amount, action, timestamp) VALUES (?, ?, ?, ?)",
        (current_user["id"], req.amount, "Manual Top-up", now_iso)
    )
    conn.commit()
    
    return {
        "success": True,
        "message": f"Successfully added {req.amount} tokens.",
        "tokens_remaining": new_balance
    }


# ============================================================
# API ENDPOINTS (PROTECTED BY TOKEN SYSTEM)
# ============================================================

@app.post("/scrape")
async def scrape_post(
    options: ScrapeOptions,
    current_user: dict = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db)
):
    # Calculate Token Cost
    base_cost = 3 if options.use_js_fallback else 1
    ai_features = ["summary", "highlights", "json", "questions", "branding"]
    ai_cost = sum(2 for f in options.formats if f.lower() in ai_features)
    total_cost = base_cost + ai_cost

    new_balance = deduct_tokens(current_user["id"], total_cost, f"Scrape: {options.url[:30]}", conn)

    data = options.model_dump() if hasattr(options, "model_dump") else options.dict()
    res = await scrape_engine(**data)
    res["tokens_deducted"] = total_cost
    res["tokens_remaining"] = new_balance
    return res


@app.post("/map")
async def map_url(
    options: MapOptions,
    current_user: dict = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db)
):
    new_balance = deduct_tokens(current_user["id"], 1, f"Map: {options.url[:30]}", conn)
    clean_url = clean_input_url(options.url)
    parsed_base = urlparse(clean_url)

    try:
        async with httpx.AsyncClient(headers=DEFAULT_HEADERS, follow_redirects=True, timeout=15.0) as client:
            res = await client.get(clean_url)
            res.raise_for_status()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Map fetch failed: {exc}")

    soup = BeautifulSoup(res.text, "html.parser")
    links = set()

    for a in soup.find_all("a", href=True):
        abs_l = absolute_url(clean_url, a["href"])
        if abs_l:
            p = urlparse(abs_l)
            if options.include_subdomains and p.hostname and parsed_base.hostname in p.hostname:
                links.add(abs_l)
            elif p.hostname == parsed_base.hostname:
                links.add(abs_l)

    return {
        "success": True,
        "url": clean_url,
        "links": list(links)[:options.limit],
        "tokens_remaining": new_balance
    }


@app.post("/crawl")
async def crawl_url(
    options: CrawlOptions,
    current_user: dict = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db)
):
    map_res = await map_url(MapOptions(url=options.url, limit=options.limit), current_user=current_user, conn=conn)
    target_urls = map_res["links"] or [clean_input_url(options.url)]

    total_cost = len(target_urls)
    new_balance = deduct_tokens(current_user["id"], total_cost, f"Crawl {len(target_urls)} pages", conn)

    scrape_opts = options.scrape_options or ScrapeOptions(url="")
    base_opts = scrape_opts.model_dump() if hasattr(scrape_opts, "model_dump") else scrape_opts.dict()

    tasks = []
    for u in target_urls:
        opts = base_opts.copy()
        opts["url"] = u
        tasks.append(scrape_engine(**opts))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    successful = [r for r in results if isinstance(r, dict) and r.get("success")]

    return {
        "success": True,
        "crawled_count": len(successful),
        "data": successful,
        "tokens_deducted": total_cost,
        "tokens_remaining": new_balance
    }


@app.post("/search")
async def search_web(
    options: SearchOptions,
    current_user: dict = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db)
):
    if not options.query.strip():
        raise HTTPException(status_code=400, detail="Search query required.")

    new_balance = deduct_tokens(current_user["id"], 1, f"Search: {options.query[:20]}", conn)

    try:
        from duckduckgo_search import DDGS
    except ImportError:
        raise HTTPException(status_code=500, detail="duckduckgo-search package missing.")

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(options.query, max_results=options.limit))
        return {
            "success": True,
            "query": options.query,
            "results": results,
            "tokens_remaining": new_balance
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Search failed: {exc}")


# ============================================================
# AGENT CLARA (CONVERSATIONAL & STRUCTURED DATA AGENT)
# ============================================================

@app.post("/agent")
async def clara_agent(
    options: AgentOptions,
    current_user: dict = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db)
):
    """Clara: Autonomous AI Agent & Conversational Assistant for ReClaire."""
    require_gemini()
    new_balance = deduct_tokens(current_user["id"], 5, f"Agent Clara: {options.prompt[:20]}", conn)

    target_urls = options.urls or []
    scraped_contexts = []

    # If seed URLs provided, scrape them asynchronously
    if target_urls:
        tasks = [scrape_engine(url=u, formats=["markdown"], only_main_content=True) for u in target_urls[:3]]
        scraped_results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in scraped_results:
            if isinstance(res, dict) and res.get("success"):
                scraped_contexts.append(f"Source: {res['url']}\n\n{res.get('markdown', '')[:10000]}")

    gathered_web_text = "\n\n---\n\n".join(scraped_contexts) if scraped_contexts else "No external URLs scraped."

    system_instruction = """
You are Clara, the resident AI Agent and Assistant for ReClaire.
ReClaire is a modern micro-SaaS content engine that parses web pages into clean Markdown, JSON, Branding profiles, and AI insights.

Your Capabilities:
1. Converse naturally with users about web URLs, research topics, or technical queries.
2. Explain ReClaire's features (FastAPI engine, Playwright JS rendering, Gemini AI transformations, Token System).
3. Synthesize gathered web pages into answers, summaries, or structured formats.

Guidelines:
- Keep answers helpful, grounded, and engaging.
- If given gathered web sources, use them directly to answer the user's questions.
- If asked for structured output (JSON or CSV), respond strictly in that format.
"""

    output_fmt = options.output_format.lower().strip()

    # 1. CSV Mode
    if output_fmt == "csv":
        prompt = f"""
{system_instruction}

Task: Fulfill user prompt and return ONLY valid raw CSV with headers. Do not use code fences.

User Prompt: {options.prompt}
Web Context:
{gathered_web_text[:20000]}
"""
        response = gemini_client.models.generate_content(model="gemini-3.6-flash", contents=prompt)
        raw_csv = re.sub(r"```(?:csv)?\s*|\s*```", "", response.text.strip(), flags=re.IGNORECASE)
        return {"agent": "Clara", "format": "csv", "data": raw_csv, "tokens_remaining": new_balance}

    # 2. JSON Mode
    elif output_fmt == "json":
        schema = options.json_schema or {"result": "string", "key_facts": "list"}
        prompt = f"""
{system_instruction}

Task: Fulfill user prompt and return valid JSON matching this schema:
{json.dumps(schema)}

User Prompt: {options.prompt}
Web Context:
{gathered_web_text[:20000]}
"""
        response = gemini_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        return {"agent": "Clara", "format": "json", "data": parse_json_response(response.text), "tokens_remaining": new_balance}

    # 3. Conversational Chat Mode (Default)
    else:
        history_text = ""
        if options.chat_history:
            history_text = "\n".join([f"{msg.role.capitalize()}: {msg.content}" for msg in options.chat_history[-6:]])

        prompt = f"""
{system_instruction}

Recent Dialogue History:
{history_text or "No prior messages."}

User Prompt: {options.prompt}

Gathered Context from Web / Seed URLs:
{gathered_web_text[:20000]}

Provide a conversational, insightful response as Clara:
"""
        response = gemini_client.models.generate_content(model="gemini-3.6-flash", contents=prompt)
        return {
            "agent": "Clara",
            "format": "chat",
            "message": response.text.strip(),
            "sources_used": target_urls,
            "tokens_remaining": new_balance
        }


# ============================================================
# SYSTEM HEALTH & ROOT ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    return {
        "name": "ReClaire API",
        "version": "2.2.0",
        "status": "online",
        "docs": "/docs",
        "endpoints": ["/auth/register", "/auth/login", "/auth/me", "/scrape", "/map", "/crawl", "/search", "/agent"]
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "gemini_configured": gemini_client is not None,
        "database": "sqlite3 connected"
    }


# ============================================================
# LOCAL SERVER LAUNCH
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
