import asyncio
import io
import os
import json
import base64
import traceback
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, urlparse
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from pypdf import PdfReader
from playwright.async_api import async_playwright

# --- Gemini LLM Setup ---
try:
    from google import genai
    from google.genai import types
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
except ImportError:
    gemini_client = None

app = FastAPI(
    title="ReClaire API",
    description="High-performance web parser converting web assets into clean Markdown, JSON, and LLM-enriched formats.",
    version="1.9.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models ---
class ScrapeOptions(BaseModel):
    url: str
    formats: List[str] = Field(
        default_factory=lambda: ["markdown"],
        description="Supported formats: 'markdown', 'html', 'raw_html', 'links', 'images', 'screenshot', 'summary', 'highlights', 'json', 'questions', 'branding'"
    )
    use_js_fallback: bool = True
    only_main_content: bool = True
    remove_tags: List[str] = Field(
        default_factory=lambda: ["script", "style", "nav", "footer", "header", "aside", "form", "iframe", "noscript"]
    )
    wait_for_selector: Optional[str] = None
    json_schema: Optional[Dict[str, Any]] = None 
    user_question: Optional[str] = Field(None, description="The specific question to ask about the page (for Q&A mode).")
    web_augmented_qa: bool = Field(True, description="Allow the LLM to use outside knowledge to supplement the page content.")

class BatchScrapeRequest(BaseModel):
    urls: List[str]
    options: Optional[ScrapeOptions] = None

# --- Helpers & Core Engines ---
def clean_input_url(raw_url: str) -> str:
    raw_url = raw_url.strip()
    if not raw_url.startswith("http://") and not raw_url.startswith("https://"):
        return "https://" + raw_url
    return raw_url

def parse_pdf_bytes(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    extracted_pages = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            extracted_pages.append(f"## Page {idx + 1}\n\n{text.strip()}")
    return "\n\n".join(extracted_pages)

async def fetch_dynamic_content(url: str, wait_for_selector: Optional[str] = None, take_screenshot: bool = False) -> dict:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={'width': 1280, 'height': 800})
        page = await context.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        
        if wait_for_selector:
            try: await page.wait_for_selector(wait_for_selector, timeout=5000)
            except Exception: pass
                
        content = await page.content()
        screenshot_b64 = None
        if take_screenshot:
            try:
                screenshot_bytes = await page.screenshot(full_page=True)
                screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
            except Exception: pass

        await browser.close()
        return {"html": content, "screenshot": screenshot_b64}

# --- LLM Generators ---
def generate_gemini_qa(markdown_text: str, user_question: str, web_augmented: bool) -> str:
    if not gemini_client:
        raise ValueError("GEMINI_API_KEY is not set.")
    
    if web_augmented:
        prompt = f"Answer the user's question. Use the provided page content as your primary source. If the content lacks context, you may augment the answer using your general web knowledge. \n\nUser Question: {user_question}\n\nPage Content:\n{markdown_text[:15000]}"
    else:
        prompt = f"Answer the user's question STRICTLY using ONLY the provided page content. If the answer is not in the text, state that it is not available on this page. \n\nUser Question: {user_question}\n\nPage Content:\n{markdown_text[:15000]}"
        
    return gemini_client.models.generate_content(model='gemini-3.6-flash', contents=prompt).text.strip()

def generate_gemini_branding(markdown_text: str, meta: dict) -> dict:
    if not gemini_client:
        raise ValueError("GEMINI_API_KEY is not set.")
    prompt = f"Analyze the content and extract the brand profile. Return a JSON object with keys: 'brand_voice', 'core_messaging', 'value_proposition', and 'target_audience'.\n\nContent:\n{markdown_text[:10000]}"
    response = gemini_client.models.generate_content(
        model='gemini-3.6-flash',
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    return json.loads(response.text)

def generate_gemini_summary(markdown_text: str) -> str:
    if not gemini_client:
        raise ValueError("GEMINI_API_KEY is not set.")
    prompt = f"Provide a concise, comprehensive summary of the following webpage content:\n\n{markdown_text[:15000]}"
    return gemini_client.models.generate_content(model='gemini-3.6-flash', contents=prompt).text.strip()

def generate_gemini_highlights(markdown_text: str) -> Any:
    if not gemini_client:
        raise ValueError("GEMINI_API_KEY is not set.")
    prompt = f"Extract 3 to 5 key bullet-point highlights or takeaways from the following webpage content. Return them as a JSON list of strings.\n\nContent:\n{markdown_text[:10000]}"
    response = gemini_client.models.generate_content(
        model='gemini-3.6-flash',
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    return json.loads(response.text)

def generate_gemini_json_schema(markdown_text: str, json_schema: Dict[str, Any]) -> dict:
    if not gemini_client:
        raise ValueError("GEMINI_API_KEY is not set.")
    prompt = f"Extract data from the following content strictly matching this JSON schema: {json.dumps(json_schema)}.\n\nContent:\n{markdown_text[:12000]}"
    response = gemini_client.models.generate_content(
        model='gemini-3.6-flash',
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    return json.loads(response.text)

# --- Core Engine ---
async def scrape_engine(
    url: str,
    use_js_fallback: bool = True,
    only_main_content: bool = True,
    remove_tags: List[str] = None,
    wait_for_selector: Optional[str] = None,
    json_schema: Optional[Dict[str, Any]] = None,
    formats: Optional[List[str]] = None,
    user_question: Optional[str] = None,
    web_augmented_qa: bool = True
) -> dict:
    if remove_tags is None:
        remove_tags = ["script", "style", "nav", "footer", "header", "aside", "form", "iframe", "noscript"]
    req_formats = [f.lower().strip() for f in (formats or ["markdown"])]

    clean_url = clean_input_url(url)
    parsed_domain = urlparse(clean_url).hostname or clean_url

    HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    result = {"success": True, "url": clean_url}
    screenshot_b64 = None
    clean_markdown = ""
    
    meta_info = {
        "domain": parsed_domain,
        "favicon": f"https://www.google.com/s2/favicons?domain={parsed_domain}&sz=64"
    }

    # 1. Fetch Content
    html_content = ""
    if clean_url.lower().endswith(".pdf"):
        async with httpx.AsyncClient(headers=HEADERS, follow_redirects=True) as client:
            res = await client.get(clean_url)
            clean_markdown = parse_pdf_bytes(res.content)
    else:
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
                response = await client.get(clean_url, headers=HEADERS)
                if response.status_code == 200:
                    html_content = response.text
                elif use_js_fallback:
                    pw_res = await fetch_dynamic_content(clean_url, wait_for_selector, take_screenshot=("screenshot" in req_formats))
                    html_content, screenshot_b64 = pw_res["html"], pw_res["screenshot"]
        except Exception:
            if use_js_fallback:
                pw_res = await fetch_dynamic_content(clean_url, wait_for_selector, take_screenshot=("screenshot" in req_formats))
                html_content, screenshot_b64 = pw_res["html"], pw_res["screenshot"]

        # 2. Parse HTML & Enforce Internal Markdown Pipeline
        if html_content:
            soup_raw = BeautifulSoup(html_content, "html.parser")
            
            # Extract Visual/Meta Assets
            og_img = soup_raw.find("meta", attrs={"property": "og:image"})
            if og_img: meta_info["hero_image"] = og_img.get("content", "")
            
            theme_color = soup_raw.find("meta", attrs={"name": "theme-color"})
            if theme_color: meta_info["theme_color"] = theme_color.get("content", "")

            title = soup_raw.title.string.strip() if soup_raw.title and soup_raw.title.string else parsed_domain
            result["title"] = title
            
            if "links" in req_formats:
                result["links"] = list({urljoin(clean_url, a["href"].strip()) for a in soup_raw.find_all("a", href=True) if not a["href"].startswith(("javascript:", "#"))})
                
            if "images" in req_formats or "branding" in req_formats:
                extracted_images = list({urljoin(clean_url, img["src"].strip()) for img in soup_raw.find_all("img", src=True) if not img["src"].startswith("data:")})
                if "images" in req_formats: result["images"] = extracted_images
                meta_info["extracted_images"] = extracted_images[:5] # Top 5 for branding context

            for tag in ["script", "style", "noscript", "iframe"] + remove_tags:
                for element in soup_raw.find_all(tag): element.decompose()
            
            target_element = soup_raw.find("main") or soup_raw.find("article") or soup_raw.body or soup_raw if only_main_content else soup_raw.body or soup_raw
            
            # MANDATORY INTERNAL PIPELINE
            clean_markdown = md(str(target_element), heading_style="ATX", strip=["img"]).strip()
            if not clean_markdown: clean_markdown = "No readable content extracted."

            if "html" in req_formats: result["html"] = str(target_element)
            if "raw_html" in req_formats: result["raw_html"] = html_content
            if "screenshot" in req_formats and screenshot_b64: result["screenshot"] = f"data:image/png;base64,{screenshot_b64}"

    result["metadata"] = meta_info
    if "markdown" in req_formats or len(req_formats) == 0:
        result["markdown"] = clean_markdown

    # 3. LLM Processing
    if clean_markdown:
        if "questions" in req_formats and user_question:
            try: result["qa_answer"] = await asyncio.to_thread(generate_gemini_qa, clean_markdown, user_question, web_augmented_qa)
            except Exception as e: result["qa_error"] = str(e)
            
        if "summary" in req_formats:
            try: result["summary"] = await asyncio.to_thread(generate_gemini_summary, clean_markdown)
            except Exception as e: result["summary_error"] = str(e)

        if "highlights" in req_formats:
            try: result["highlights"] = await asyncio.to_thread(generate_gemini_highlights, clean_markdown)
            except Exception as e: result["highlights_error"] = str(e)

        if "json" in req_formats:
            try:
                schema = json_schema or {"title": "str", "key_takeaways": "list"}
                result["json_data"] = await asyncio.to_thread(generate_gemini_json_schema, clean_markdown, schema)
            except Exception as e: result["json_error"] = str(e)

        if "branding" in req_formats:
            try:
                llm_brand = await asyncio.to_thread(generate_gemini_branding, clean_markdown, meta_info)
                result["branding"] = {
                    "visual_assets": meta_info,
                    "brand_analysis": llm_brand
                }
            except Exception as e: result["branding_error"] = str(e)

    return result

# --- API Endpoints ---
@app.post("/scrape")
async def scrape_post(options: ScrapeOptions):
    return await scrape_engine(**options.dict())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
