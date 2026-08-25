import asyncio
import io
import os
import json
import traceback  # Robust error logging
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
    description="High-performance web parser & crawler converting web assets into clean Markdown and Structured JSON.",
    version="1.6.0"
)

# Robust CORS Configuration to allow cross-origin requests from any frontend
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
    use_js_fallback: bool = True
    only_main_content: bool = True
    remove_tags: List[str] = Field(
        default_factory=lambda: ["script", "style", "nav", "footer", "header", "aside", "form", "iframe", "noscript"]
    )
    wait_for_selector: Optional[str] = None
    json_schema: Optional[Dict[str, Any]] = None 

class BatchScrapeRequest(BaseModel):
    urls: List[str]
    options: Optional[ScrapeOptions] = None

class MapRequest(BaseModel):
    url: str
    limit: int = 100


# --- Helpers & Core Engines ---

def clean_input_url(raw_url: str) -> str:
    raw_url = raw_url.strip()
    if "](" in raw_url:
        parts = raw_url.split("](", 1)
        url_part = parts[1]
        if url_part.endswith(")"):
            url_part = url_part[:-1]
        return url_part.strip()
    if raw_url.startswith("[") and raw_url.endswith("]"):
        return raw_url[1:-1].strip()
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

async def fetch_dynamic_html(url: str, wait_for_selector: Optional[str] = None) -> str:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            locale="en-US"
        )
        page = await context.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        
        if wait_for_selector:
            try:
                await page.wait_for_selector(wait_for_selector, timeout=5000)
            except Exception as e:
                print(f"Playwright selector wait timeout:\n{traceback.format_exc()}")
                
        content = await page.content()
        await browser.close()
        return content

def generate_gemini_json(markdown_text: str, schema: dict) -> dict:
    if not gemini_client:
        raise ValueError("GEMINI_API_KEY is not set or google-genai is not installed.")
    
    prompt = f"You are a precise data extraction assistant. Extract data from the following markdown content matching the requested JSON structure.\n\nSchema:\n{json.dumps(schema, indent=2)}\n\nMarkdown Content:\n{markdown_text}"
    
    response = gemini_client.models.generate_content(
        model='gemini-3.6-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        )
    )
    return json.loads(response.text)

async def scrape_engine(
    url: str,
    use_js_fallback: bool = True,
    only_main_content: bool = True,
    remove_tags: List[str] = None,
    wait_for_selector: Optional[str] = None,
    json_schema: Optional[Dict[str, Any]] = None
) -> dict:
    if remove_tags is None:
        remove_tags = ["script", "style", "nav", "footer", "header", "aside", "form", "iframe", "noscript"]

    clean_url = clean_input_url(url)
    parsed_domain = urlparse(clean_url).hostname or clean_url

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webkit,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Sec-Ch-Ua": '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1"
    }

    result = {}

    # 1. Handle PDF Documents
    if clean_url.lower().endswith(".pdf"):
        try:
            async with httpx.AsyncClient(headers=HEADERS, follow_redirects=True) as client:
                res = await client.get(clean_url)
                if res.status_code == 403:
                    raise HTTPException(status_code=403, detail="Access forbidden by target PDF server.")
                res.raise_for_status()
                pdf_md = parse_pdf_bytes(res.content)
                result = {
                    "success": True,
                    "url": clean_url,
                    "title": clean_url.split("/")[-1],
                    "markdown": pdf_md,
                    "metadata": {
                        "domain": parsed_domain,
                        "favicon": f"https://www.google.com/s2/favicons?domain={parsed_domain}&sz=64"
                    }
                }
        except HTTPException:
            raise
        except Exception as e:
            print(f"PDF extraction error:\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"PDF extraction error: {str(e)}")
    
    else:
        # 2. Web Scraping with httpx + Playwright Fallback
        html_content = ""
        used_fallback = False

        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
                response = await client.get(clean_url, headers=HEADERS)
                
                if "application/pdf" in response.headers.get("content-type", "").lower():
                    pdf_md = parse_pdf_bytes(response.content)
                    result = {
                        "success": True,
                        "url": clean_url,
                        "title": clean_url.split("/")[-1],
                        "markdown": pdf_md,
                        "metadata": {
                            "domain": parsed_domain,
                            "favicon": f"https://www.google.com/s2/favicons?domain={parsed_domain}&sz=64"
                        }
                    }
                elif response.status_code == 200:
                    html_content = response.text
                elif use_js_fallback:
                    html_content = await fetch_dynamic_html(clean_url, wait_for_selector)
                    used_fallback = True
                else:
                    raise HTTPException(status_code=400, detail=f"HTTP Error {response.status_code}")
        except Exception as py_err:
            if use_js_fallback:
                try:
                    html_content = await fetch_dynamic_html(clean_url, wait_for_selector)
                    used_fallback = True
                except Exception as js_err:
                    print(f"JS Fallback Scraping failed:\n{traceback.format_exc()}")
                    raise HTTPException(status_code=500, detail=f"Scraping failed: {str(js_err)}")
            else:
                print(f"Network connection failed:\n{traceback.format_exc()}")
                raise HTTPException(status_code=500, detail="Network connection failed.")

        if use_js_fallback and not used_fallback and not result and len(html_content.strip()) < 300:
            try:
                html_content = await fetch_dynamic_html(clean_url, wait_for_selector)
            except Exception as e:
                print(f"Secondary JS Fallback failed:\n{traceback.format_exc()}")

        # 3. HTML Cleanup & Markdown Parsing 
        if not result:
            try:
                soup_base = BeautifulSoup(html_content, "html.parser")
                
                # Base cleanup: strip scripts and styles
                for tag in ["script", "style", "noscript", "iframe"]:
                    for element in soup_base.find_all(tag):
                        element.decompose()

                # Attempt aggressive tag cleanup on a working copy
                soup_working = BeautifulSoup(str(soup_base), "html.parser")
                additional_tags = [t for t in remove_tags if t not in ["script", "style", "noscript", "iframe"]]
                for tag in additional_tags:
                    for element in soup_working.find_all(tag):
                        element.decompose()

                # Fallback to base soup if aggressive cleanup stripped essential page content (e.g., Google homepage)
                if len(soup_working.get_text(strip=True)) > 30:
                    soup_final = soup_working
                else:
                    soup_final = soup_base

                title = soup_final.title.string.strip() if soup_final.title and soup_final.title.string else parsed_domain
                
                if only_main_content:
                    target_element = (
                        soup_final.find("main") or 
                        soup_final.find("article") or 
                        soup_final.find("div", {"id": "content"}) or 
                        soup_final.find("div", {"id": "bodyContent"}) or 
                        soup_final.find("div", {"id": "main-content"}) or 
                        soup_final.find("div", {"id": "main"}) or
                        soup_final.body or 
                        soup_final
                    )
                else:
                    target_element = soup_final.body or soup_final

                raw_html = str(target_element) if target_element else str(soup_final)
                clean_markdown = md(raw_html, heading_style="ATX", strip=["img"]).strip()
                
                if not clean_markdown:
                    clean_markdown = "No readable content could be extracted from this webpage."
                    
                result = {
                    "success": True,
                    "url": clean_url,
                    "title": title,
                    "markdown": clean_markdown,
                    "metadata": {
                        "domain": parsed_domain,
                        "favicon": f"https://www.google.com/s2/favicons?domain={parsed_domain}&sz=64"
                    }
                }
            except Exception as e:
                print(f"BeautifulSoup Parsing error:\n{traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=f"Parsing error: {str(e)}")

    # 4. JSON Schema Extraction Step 
    if json_schema and result.get("success"):
        try:
            extracted_json = await asyncio.to_thread(generate_gemini_json, result["markdown"], json_schema)
            result["json_data"] = extracted_json
        except Exception as e:
            print(f"Gemini JSON Extraction error:\n{traceback.format_exc()}")
            result["json_data_error"] = f"JSON extraction failed: {str(e)}"

    return result


# --- API Endpoints ---

@app.get("/")
def home():
    return {"status": "online", "engine": "ReClaire v1.6.0"}

@app.get("/scrape")
async def scrape_get(url: str = Query(..., description="Target URL")):
    try:
        return await scrape_engine(url=url)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unhandled error in GET /scrape:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/scrape")
async def scrape_post(options: ScrapeOptions):
    try:
        return await scrape_engine(
            url=options.url,
            use_js_fallback=options.use_js_fallback,
            only_main_content=options.only_main_content,
            remove_tags=options.remove_tags,
            wait_for_selector=options.wait_for_selector,
            json_schema=options.json_schema
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unhandled error in POST /scrape:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/batch")
async def batch_scrape(payload: BatchScrapeRequest):
    semaphore = asyncio.Semaphore(5)

    async def worker(target_url: str):
        async with semaphore:
            try:
                opts = payload.options or ScrapeOptions(url=target_url)
                return await scrape_engine(
                    url=target_url,
                    use_js_fallback=opts.use_js_fallback,
                    only_main_content=opts.only_main_content,
                    remove_tags=opts.remove_tags,
                    wait_for_selector=opts.wait_for_selector,
                    json_schema=opts.json_schema 
                )
            except Exception as e:
                print(f"Error in batch worker for URL {target_url}:\n{traceback.format_exc()}")
                return {"success": False, "url": target_url, "error": str(e)}

    results = await asyncio.gather(*(worker(u) for u in payload.urls))
    return {
        "success": True,
        "total": len(payload.urls),
        "data": results
    }

@app.post("/map")
async def map_domain(payload: MapRequest):
    try:
        clean_url = clean_input_url(payload.url)
        parsed = urlparse(clean_url)
        domain_root = f"{parsed.scheme}://{parsed.netloc}"
        found_urls = set()
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

        sitemap_url = urljoin(domain_root, "/sitemap.xml")
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
                res = await client.get(sitemap_url, headers=headers)
                if res.status_code == 200:
                    root = ET.fromstring(res.text)
                    for elem in root.iter():
                        if elem.tag.endswith("loc") and elem.text:
                            found_urls.add(elem.text.strip())
        except Exception:
            pass

        if not found_urls:
            try:
                async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
                    res = await client.get(clean_url, headers=headers)
                    if res.status_code == 200:
                        soup = BeautifulSoup(res.text, "html.parser")
                        for tag in soup.find_all("a", href=True):
                            full_link = urljoin(clean_url, tag["href"])
                            if urlparse(full_link).netloc == parsed.netloc:
                                found_urls.add(full_link)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Mapping failed: {str(e)}")

        url_list = list(found_urls)[:payload.limit]
        return {
            "success": True,
            "domain": domain_root,
            "count": len(url_list),
            "urls": url_list
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unhandled error in POST /map:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
