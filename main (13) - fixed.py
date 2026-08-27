import asyncio
import io
import os
import json
import base64
import re
import traceback
from urllib.parse import urljoin, urlparse

from typing import List, Optional, Dict, Any

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from pypdf import PdfReader
from playwright.async_api import async_playwright

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# GEMINI SETUP
# ============================================================

try:
    from google import genai
    from google.genai import types

    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

    gemini_client = (
        genai.Client(api_key=GEMINI_API_KEY)
        if GEMINI_API_KEY
        else None
    )

except ImportError:
    genai = None
    types = None
    gemini_client = None


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="ReClaire API",
    description=(
        "High-performance web parser converting web assets into "
        "clean Markdown, JSON, images, branding, and LLM-enriched formats."
    ),
    version="2.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DATA MODELS
# ============================================================

class ScrapeOptions(BaseModel):
    url: str

    formats: List[str] = Field(
        default_factory=lambda: ["markdown"],
        description=(
            "Supported formats: markdown, html, raw_html, links, "
            "images, screenshot, summary, highlights, json, "
            "questions, branding"
        ),
    )

    use_js_fallback: bool = True

    only_main_content: bool = True

    remove_tags: List[str] = Field(
        default_factory=lambda: [
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "form",
            "iframe",
            "noscript",
        ]
    )

    wait_for_selector: Optional[str] = None

    json_schema: Optional[Dict[str, Any]] = None

    # IMPORTANT:
    # This matches the frontend/API contract.
    user_question: Optional[str] = Field(
        None,
        description="Question to answer using the scraped page.",
    )

    web_augmented_qa: bool = Field(
        True,
        description=(
            "Allow the model to use general knowledge when "
            "the page does not contain enough context."
        ),
    )


class BatchScrapeRequest(BaseModel):
    urls: List[str]
    options: Optional[ScrapeOptions] = None

class MapOptions(BaseModel):
    url: str
    limit: int = Field(100, ge=1, le=500, description="Maximum number of links to map.")
    include_subdomains: bool = False

class CrawlOptions(BaseModel):
    url: str
    limit: int = Field(5, ge=1, le=25, description="Maximum number of pages to crawl.")
    scrape_options: Optional[ScrapeOptions] = None

class SearchOptions(BaseModel):
    query: str
    limit: int = Field(5, ge=1, le=25, description="Number of links to return.")

class AgentOptions(BaseModel):
    prompt: str
    urls: Optional[List[str]] = Field(None, description="Provide seed URLs, otherwise Clara searches the web automatically.")
    output_format: str = Field("json", description="Choose 'json' or 'csv'")
    json_schema: Optional[Dict[str, Any]] = None


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,"
        "application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def clean_input_url(raw_url: str) -> str:
    """
    Normalize a URL supplied by the user.
    """

    if not raw_url:
        return ""

    raw_url = raw_url.strip().strip("<>")

    if not raw_url:
        return ""

    if not raw_url.startswith(("http://", "https://")):
        raw_url = "https://" + raw_url

    return raw_url


def safe_text(value: Any) -> str:
    """
    Safely normalize text extracted from HTML.
    Handles None, strings, and lists (e.g., HTML class attributes).
    """
    if not value:
        return ""

    # BeautifulSoup returns lists for multi-valued attributes like 'class'
    if isinstance(value, list):
        value = " ".join(str(v) for v in value)
    else:
        value = str(value)

    return re.sub(r"\s+", " ", value).strip()


def unique_list(values: List[str]) -> List[str]:
    """
    Preserve order while removing duplicates.
    """

    seen = set()
    result = []

    for value in values:
        if not value:
            continue

        if value not in seen:
            seen.add(value)
            result.append(value)

    return result


def absolute_url(base_url: str, value: Optional[str]) -> Optional[str]:
    """
    Convert a relative URL to an absolute URL.
    """

    if not value:
        return None

    value = value.strip()

    if not value:
        return None

    # Ignore data/blob/javascript URLs.
    if value.startswith(
        (
            "data:",
            "blob:",
            "javascript:",
            "mailto:",
            "tel:",
        )
    ):
        return None

    return urljoin(base_url, value)


def is_http_url(value: Optional[str]) -> bool:
    if not value:
        return False
    try:
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.hostname)
    except Exception:
        return False


def same_domain(host_a: Optional[str], host_b: Optional[str]) -> bool:
    if not host_a or not host_b:
        return False
    return host_a.lower().rstrip(".") == host_b.lower().rstrip(".")


def parse_json_response(text: str) -> Any:
    """
    Parse Gemini JSON output safely.

    Gemini normally returns clean JSON when response_mime_type is used,
    but this also handles accidental markdown code fences.
    """

    if not text:
        raise ValueError("Gemini returned an empty response.")

    text = text.strip()

    # Remove ```json ... ``` wrappers if present.
    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

    return json.loads(text)


# ============================================================
# PDF
# ============================================================

def parse_pdf_bytes(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))

    extracted_pages = []

    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""

        if text.strip():
            extracted_pages.append(
                f"## Page {idx + 1}\n\n{text.strip()}"
            )

    return "\n\n".join(extracted_pages)


# ============================================================
# PLAYWRIGHT
# ============================================================

async def fetch_dynamic_content(
    url: str,
    wait_for_selector: Optional[str] = None,
    take_screenshot: bool = False,
) -> dict:

    async with async_playwright() as p:

        browser = await p.chromium.launch(
            headless=True
        )

        context = await browser.new_context(
            viewport={
                "width": 1280,
                "height": 800,
            },
            user_agent=DEFAULT_HEADERS["User-Agent"],
        )

        page = await context.new_page()

        try:

            await page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=30000,
            )

            # Give JS applications a short opportunity to render.
            try:
                await page.wait_for_load_state(
                    "networkidle",
                    timeout=8000,
                )
            except Exception:
                pass

            if wait_for_selector:

                try:
                    await page.wait_for_selector(
                        wait_for_selector,
                        timeout=5000,
                    )
                except Exception:
                    pass

            content = await page.content()

            screenshot_b64 = None

            if take_screenshot:

                try:
                    screenshot_bytes = await page.screenshot(
                        full_page=True
                    )

                    screenshot_b64 = base64.b64encode(
                        screenshot_bytes
                    ).decode("utf-8")

                except Exception:
                    screenshot_b64 = None

            return {
                "html": content,
                "screenshot": screenshot_b64,
            }

        finally:
            await browser.close()


# ============================================================
# IMAGE EXTRACTION
# ============================================================

def _srcset_urls(value: Optional[str], base_url: str) -> List[str]:
    if not value:
        return []

    urls = []
    for candidate in str(value).split(","):
        candidate = candidate.strip()
        if not candidate:
            continue
        url = absolute_url(base_url, candidate.split()[0])
        if url and is_http_url(url):
            urls.append(url)
    return urls


def _image_candidates(img, base_url: str) -> List[str]:
    candidates = []

    for attr in (
        "src",
        "data-src",
        "data-lazy-src",
        "data-original",
        "data-image",
        "data-url",
        "data-lazy",
        "data-fallback-src",
    ):
        value = img.get(attr)
        if value:
            url = absolute_url(base_url, value)
            if url and is_http_url(url):
                candidates.append(url)

    candidates.extend(_srcset_urls(img.get("srcset"), base_url))
    candidates.extend(_srcset_urls(img.get("data-srcset"), base_url))
    return unique_list(candidates)


def extract_image_urls(soup: BeautifulSoup, base_url: str) -> List[str]:
    images = []

    for img in soup.find_all("img"):
        images.extend(_image_candidates(img, base_url))

    for source in soup.find_all("source"):
        images.extend(_srcset_urls(source.get("srcset"), base_url))
        images.extend(_srcset_urls(source.get("data-srcset"), base_url))

        src = absolute_url(base_url, source.get("src"))
        if src and is_http_url(src):
            images.append(src)

    return unique_list(images)


def extract_image_metadata(
    soup: BeautifulSoup,
    base_url: str,
) -> List[Dict[str, Any]]:
    results = []

    for img in soup.find_all("img"):
        candidates = _image_candidates(img, base_url)
        if not candidates:
            continue

        results.append({
            "url": candidates[0],
            "src_candidates": candidates[:6],
            "alt": safe_text(img.get("alt")),
            "title": safe_text(img.get("title")),
            "width": img.get("width"),
            "height": img.get("height"),
            "loading": safe_text(img.get("loading")),
            "class": safe_text(img.get("class")),
            "id": safe_text(img.get("id")),
        })

    seen = set()
    cleaned = []
    for item in results:
        if item["url"] in seen:
            continue
        seen.add(item["url"])
        cleaned.append(item)

    return cleaned


# ============================================================
# BRANDING EXTRACTION
# ============================================================

def extract_branding_assets(
    soup: BeautifulSoup,
    base_url: str,
    domain: str,
) -> Dict[str, Any]:
    assets = {
        "domain": domain,
        "site_name": None,
        "title": None,
        "description": None,
        "favicon": None,
        "apple_touch_icon": None,
        "logo": None,
        "logo_type": None,
        "logo_candidates": [],
        "hero_image": None,
        "og_image": None,
        "twitter_image": None,
        "theme_color": None,
        "manifest": None,
        "fonts": [],
        "css_variables": {},
        "meta": {},
    }

    if soup.title:
        assets["title"] = safe_text(soup.title.get_text())

    def get_meta(*, name: Optional[str] = None,
                 property_name: Optional[str] = None) -> Optional[str]:
        tag = (
            soup.find("meta", attrs={"name": name})
            if name
            else soup.find("meta", attrs={"property": property_name})
        )
        return safe_text(tag.get("content")) if tag else None

    assets["description"] = (
        get_meta(name="description")
        or get_meta(property_name="og:description")
    )
    assets["site_name"] = (
        get_meta(property_name="og:site_name")
        or get_meta(name="application-name")
    )
    assets["og_image"] = absolute_url(base_url, get_meta(property_name="og:image"))
    assets["hero_image"] = assets["og_image"]
    assets["twitter_image"] = absolute_url(base_url, get_meta(name="twitter:image"))
    assets["theme_color"] = (
        get_meta(name="theme-color")
        or get_meta(name="msapplication-TileColor")
    )

    # Icons
    icon_candidates = []
    for link in soup.find_all("link"):
        rel = link.get("rel", [])
        if isinstance(rel, str):
            rel = [rel]
        rel_lower = {str(item).lower() for item in rel}
        href = link.get("href")
        if not href:
            continue

        if "icon" in rel_lower or "shortcut icon" in rel_lower or "apple-touch-icon" in rel_lower:
            icon_url = absolute_url(base_url, href)
            if not icon_url or not is_http_url(icon_url):
                continue

            if "apple-touch-icon" in rel_lower:
                assets["apple_touch_icon"] = icon_url

            icon_candidates.append({
                "url": icon_url,
                "rel": sorted(rel_lower),
                "sizes": link.get("sizes"),
                "type": link.get("type"),
            })

    if icon_candidates:
        assets["favicon"] = assets["apple_touch_icon"] or icon_candidates[0]["url"]

    if not assets["favicon"]:
        assets["favicon"] = (
            f"https://www.google.com/s2/favicons?domain={domain}&sz=64"
        )

    # Manifest
    manifest_link = soup.find(
        "link",
        rel=lambda value: (
            value
            and (
                "manifest" in value
                if isinstance(value, str)
                else "manifest" in value
            )
        ),
    )
    if manifest_link:
        assets["manifest"] = absolute_url(base_url, manifest_link.get("href"))

    # Logo candidates
    logo_candidates = []

    for img in soup.find_all("img"):
        combined = " ".join([
            safe_text(img.get("alt")),
            safe_text(img.get("class")),
            safe_text(img.get("id")),
        ]).lower()

        if "logo" not in combined:
            continue

        candidates = _image_candidates(img, base_url)
        if candidates:
            logo_candidates.append({
                "url": candidates[0],
                "type": "image",
                "alt": safe_text(img.get("alt")),
                "width": img.get("width"),
                "height": img.get("height"),
            })

    for svg in soup.find_all("svg"):
        combined = " ".join([
            safe_text(svg.get("id")),
            safe_text(svg.get("class")),
            safe_text(svg.get("aria-label")),
            safe_text(svg.get("role")),
        ]).lower()

        if "logo" in combined:
            markup = str(svg)
            if len(markup) <= 200_000:
                logo_candidates.append({
                    "type": "inline-svg",
                    "markup": markup,
                    "id": safe_text(svg.get("id")),
                    "aria_label": safe_text(svg.get("aria-label")),
                })

    assets["logo_candidates"] = logo_candidates
    if logo_candidates:
        first = logo_candidates[0]
        assets["logo"] = first.get("url") or first.get("markup")
        assets["logo_type"] = first.get("type")

    # Fonts
    font_values = []
    for style in soup.find_all("style"):
        css = style.get_text(" ", strip=True)
        if not css:
            continue
        for match in re.findall(r"font-family\s*:\s*([^;}{]+)", css, flags=re.IGNORECASE):
            cleaned = safe_text(match)
            if cleaned:
                font_values.append(cleaned)
    assets["fonts"] = unique_list(font_values)[:20]

    # CSS variables
    css_variables = {}
    for style in soup.find_all("style"):
        css = style.get_text(" ", strip=True)
        if not css:
            continue
        for key, value in re.findall(r"--([a-zA-Z0-9_-]+)\s*:\s*([^;}{]+)", css):
            key, value = safe_text(key), safe_text(value)
            if key and value:
                css_variables[f"--{key}"] = value
    assets["css_variables"] = dict(list(css_variables.items())[:50])

    for key in ("author", "keywords", "application-name", "theme-color", "generator", "viewport"):
        value = get_meta(name=key)
        if value:
            assets["meta"][key] = value

    return assets


# ============================================================
# GEMINI
# ============================================================

def require_gemini():
    if not gemini_client:
        raise ValueError(
            "GEMINI_API_KEY is not configured on the server."
        )


def generate_gemini_qa(
    markdown_text: str,
    user_question: str,
    web_augmented: bool,
) -> str:

    require_gemini()

    if web_augmented:

        prompt = f"""
Answer the user's question using the supplied webpage content
as the primary source.

If the page does not contain enough information, you may use
your general knowledge to provide useful context.

Do not claim that information came from the webpage if it
was not actually present there.

User Question:
{user_question}

Webpage Content:
{markdown_text[:20000]}
"""

    else:

        prompt = f"""
Answer the user's question STRICTLY using ONLY the supplied
webpage content.

If the answer cannot be found in the webpage content, say:
"The answer is not available in the extracted page content."

Do not invent information.

User Question:
{user_question}

Webpage Content:
{markdown_text[:20000]}
"""

    response = gemini_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    if not response.text:
        raise ValueError(
            "Gemini returned an empty Q&A response."
        )

    return response.text.strip()


def generate_gemini_branding(
    markdown_text: str,
    branding_assets: Dict[str, Any],
) -> Dict[str, Any]:

    require_gemini()

    prompt = f"""
Analyze the webpage and its extracted visual metadata to create
a useful brand identity profile.

Return ONLY valid JSON.

Use this exact structure:

{{
  "brand_name": "...",
  "brand_voice": "...",
  "core_messaging": "...",
  "value_proposition": "...",
  "target_audience": "...",
  "visual_style": "...",
  "color_palette": [],
  "typography": [],
  "logo_observations": "...",
  "design_observations": "..."
}}

Important:
- Do not invent exact colors unless they are supported by the
  supplied metadata/content.
- If something cannot be determined, use null, an empty list,
  or "Not detected".
- Distinguish between information directly detected and
  reasonable interpretation.

Extracted Visual Metadata:
{json.dumps(branding_assets, ensure_ascii=False, indent=2)[:20000]}

Page Content:
{markdown_text[:12000]}
"""

    response = gemini_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        ),
    )

    return parse_json_response(
        response.text
    )


def generate_gemini_summary(
    markdown_text: str,
) -> str:

    require_gemini()

    prompt = f"""
Provide a concise but comprehensive summary of this webpage.

Webpage:
{markdown_text[:20000]}
"""

    response = gemini_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    return (
        response.text.strip()
        if response.text
        else ""
    )


def generate_gemini_highlights(
    markdown_text: str,
) -> Any:

    require_gemini()

    prompt = f"""
Extract 3 to 5 important takeaways from the webpage.

Return ONLY a JSON array of strings.

Webpage:
{markdown_text[:15000]}
"""

    response = gemini_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        ),
    )

    return parse_json_response(
        response.text
    )


def generate_gemini_json_schema(
    markdown_text: str,
    json_schema: Dict[str, Any],
) -> dict:

    require_gemini()

    prompt = f"""
Extract data from the webpage.

The returned JSON MUST follow this schema:

{json.dumps(json_schema, ensure_ascii=False)}

Webpage:
{markdown_text[:15000]}
"""

    response = gemini_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        ),
    )

    return parse_json_response(
        response.text
    )


# ============================================================
# CORE SCRAPER
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

        remove_tags = [
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "form",
            "iframe",
            "noscript",
        ]

    req_formats = [
        str(f).lower().strip()
        for f in (formats or ["markdown"])
    ]

    # Remove duplicates while preserving order.
    req_formats = list(
        dict.fromkeys(req_formats)
    )

    clean_url = clean_input_url(url)

    if not clean_url:
        raise HTTPException(
            status_code=400,
            detail="A URL is required.",
        )

    parsed = urlparse(clean_url)

    if not parsed.hostname:
        raise HTTPException(
            status_code=400,
            detail="Invalid URL.",
        )

    parsed_domain = parsed.hostname

    result = {
        "success": True,
        "url": clean_url,
    }

    screenshot_b64 = None
    clean_markdown = ""
    html_content = ""

    # --------------------------------------------------------
    # Initial metadata
    # --------------------------------------------------------

    meta_info = {
        "domain": parsed_domain,
        "favicon": (
            f"https://www.google.com/s2/favicons"
            f"?domain={parsed_domain}&sz=64"
        ),
    }

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    if clean_url.lower().split("?")[0].endswith(".pdf"):

        try:

            async with httpx.AsyncClient(
                headers=DEFAULT_HEADERS,
                follow_redirects=True,
                timeout=30.0,
            ) as client:

                response = await client.get(
                    clean_url
                )

                response.raise_for_status()

                clean_markdown = parse_pdf_bytes(
                    response.content
                )

        except Exception as exc:

            raise HTTPException(
                status_code=502,
                detail=f"Failed to download PDF: {exc}",
            )

    # --------------------------------------------------------
    # HTML
    # --------------------------------------------------------

    else:

        # ----------------------------------------------------
        # First attempt: normal HTTP request
        # ----------------------------------------------------

        try:

            async with httpx.AsyncClient(
                headers=DEFAULT_HEADERS,
                follow_redirects=True,
                timeout=20.0,
            ) as client:

                response = await client.get(
                    clean_url
                )

                if response.status_code == 200:

                    content_type = (
                        response.headers.get(
                            "content-type",
                            ""
                        ).lower()
                    )

                    if (
                        "text/html" in content_type
                        or "application/xhtml+xml"
                        in content_type
                        or not content_type
                    ):
                        html_content = response.text

                # If normal HTTP failed, try Playwright.
                elif use_js_fallback:

                    pw_res = await fetch_dynamic_content(
                        clean_url,
                        wait_for_selector,
                        take_screenshot=(
                            "screenshot" in req_formats
                        ),
                    )

                    html_content = pw_res["html"]
                    screenshot_b64 = pw_res[
                        "screenshot"
                    ]

        except Exception:

            if use_js_fallback:

                try:

                    pw_res = await fetch_dynamic_content(
                        clean_url,
                        wait_for_selector,
                        take_screenshot=(
                            "screenshot" in req_formats
                        ),
                    )

                    html_content = pw_res["html"]
                    screenshot_b64 = pw_res[
                        "screenshot"
                    ]

                except Exception as exc:

                    raise HTTPException(
                        status_code=502,
                        detail=(
                            "Unable to retrieve webpage. "
                            f"HTTP and JavaScript rendering failed: {exc}"
                        ),
                    )

        # ----------------------------------------------------
        # If nothing was obtained
        # ----------------------------------------------------

        if not html_content:

            raise HTTPException(
                status_code=502,
                detail=(
                    "The webpage could not be retrieved. "
                    "The server returned no usable HTML. "
                    "Try enabling JS Rendering for JavaScript-heavy sites."
                ),
            )

        # ----------------------------------------------------
        # Parse HTML
        # ----------------------------------------------------

        soup_raw = BeautifulSoup(
            html_content,
            "html.parser",
        )

        # ----------------------------------------------------
        # Branding / metadata BEFORE destructive cleaning
        # ----------------------------------------------------

        try:
            branding_assets = extract_branding_assets(
                soup_raw,
                clean_url,
                parsed_domain,
            )
        except Exception as e:
            # Fault tolerance: Fallback to basic domain info without crashing
            print(f"Branding extraction failed for {clean_url}: {e}")
            branding_assets = {
                "domain": parsed_domain,
                "title": parsed_domain,
                "meta": {}
            }

        meta_info.update(
            branding_assets
        )

        result["title"] = (
            branding_assets.get("title")
            or parsed_domain
        )

        # ----------------------------------------------------
        # Links
        # ----------------------------------------------------

        if "links" in req_formats:

            links = []

            for a in soup_raw.find_all(
                "a",
                href=True,
            ):

                href = a.get("href", "").strip()

                if not href:
                    continue

                if href.startswith(
                    (
                        "javascript:",
                        "#",
                        "mailto:",
                        "tel:",
                    )
                ):
                    continue

                absolute = absolute_url(
                    clean_url,
                    href,
                )

                if absolute:
                    links.append(absolute)

            result["links"] = unique_list(
                links
            )

        # ----------------------------------------------------
        # Images
        # ----------------------------------------------------

        image_metadata = extract_image_metadata(
            soup_raw,
            clean_url,
        )

        image_urls = [
            item["url"]
            for item in image_metadata
        ]

        if (
            "images" in req_formats
            or "branding" in req_formats
        ):

            if "images" in req_formats:

                result["images"] = image_metadata

            meta_info[
                "extracted_images"
            ] = image_metadata[:10]

            meta_info[
                "image_count"
            ] = len(image_metadata)

        # ----------------------------------------------------
        # Remove unwanted elements from a COPY
        # ----------------------------------------------------
        #
        # IMPORTANT:
        # We intentionally don't mutate the original soup.
        # This keeps metadata/assets available.
        # ----------------------------------------------------

        soup_content = BeautifulSoup(
            str(soup_raw),
            "html.parser",
        )

        for tag in set(
            [
                "script",
                "style",
                "noscript",
                "iframe",
            ] + remove_tags
        ):

            for element in soup_content.find_all(tag):

                try:
                    element.decompose()
                except Exception:
                    pass

        # ----------------------------------------------------
        # Select content
        # ----------------------------------------------------

        if only_main_content:

            target_element = (
                soup_content.find("main")
                or soup_content.find("article")
                or soup_content.body
                or soup_content
            )

        else:

            target_element = (
                soup_content.body
                or soup_content
            )

        # ----------------------------------------------------
        # Markdown
        # ----------------------------------------------------
        #
        # IMPORTANT:
        # DO NOT strip img tags.
        # markdownify will convert them to:
        #
        # ![alt](image-url)
        #
        # ----------------------------------------------------

        clean_markdown = md(
            str(target_element),
            heading_style="ATX",
        ).strip()

        if not clean_markdown:

            clean_markdown = (
                "No readable content extracted."
            )

        # ----------------------------------------------------
        # Optional HTML outputs
        # ----------------------------------------------------

        if "html" in req_formats:

            result["html"] = str(
                target_element
            )

        if "raw_html" in req_formats:

            result["raw_html"] = html_content

        if (
            "screenshot" in req_formats
            and screenshot_b64
        ):

            result["screenshot"] = (
                "data:image/png;base64,"
                + screenshot_b64
            )

    # ========================================================
    # COMMON RESULT METADATA
    # ========================================================

    result["metadata"] = meta_info

    if (
        "markdown" in req_formats
        or len(req_formats) == 0
    ):

        result["markdown"] = clean_markdown

    # ========================================================
    # LLM PROCESSING
    # ========================================================

    if clean_markdown:

        # ----------------------------------------------------
        # Q&A
        # ----------------------------------------------------

        if (
            "questions" in req_formats
            and user_question
            and user_question.strip()
        ):

            try:

                result["qa_answer"] = (
                    await asyncio.to_thread(
                        generate_gemini_qa,
                        clean_markdown,
                        user_question.strip(),
                        web_augmented_qa,
                    )
                )

            except Exception as exc:

                result["qa_error"] = str(exc)

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        if "summary" in req_formats:

            try:

                result["summary"] = (
                    await asyncio.to_thread(
                        generate_gemini_summary,
                        clean_markdown,
                    )
                )

            except Exception as exc:

                result["summary_error"] = str(exc)

        # ----------------------------------------------------
        # Highlights
        # ----------------------------------------------------

        if "highlights" in req_formats:

            try:

                result["highlights"] = (
                    await asyncio.to_thread(
                        generate_gemini_highlights,
                        clean_markdown,
                    )
                )

            except Exception as exc:

                result["highlights_error"] = str(exc)

        # ----------------------------------------------------
        # Custom JSON
        # ----------------------------------------------------

        if "json" in req_formats:

            try:

                schema = (
                    json_schema
                    or {
                        "title": "str",
                        "key_takeaways": "list",
                    }
                )

                result["json_data"] = (
                    await asyncio.to_thread(
                        generate_gemini_json_schema,
                        clean_markdown,
                        schema,
                    )
                )

            except Exception as exc:

                result["json_error"] = str(exc)

        # ----------------------------------------------------
        # Branding
        # ----------------------------------------------------

        if "branding" in req_formats:

            try:

                brand_analysis = (
                    await asyncio.to_thread(
                        generate_gemini_branding,
                        clean_markdown,
                        meta_info,
                    )
                )

                result["branding"] = {
                    "visual_assets": meta_info,
                    "brand_analysis": brand_analysis,
                }

            except Exception as exc:

                result["branding_error"] = str(exc)

    # ========================================================
    # FINAL RESPONSE INFORMATION
    # ========================================================

    result["requested_formats"] = req_formats

    return result

# ============================================================
# MAP, CRAWL, SEARCH & AGENT (CLARA)
# ============================================================

@app.post("/map")
async def map_url(options: MapOptions):
    """
    Crawls the target URL and maps out internal links.
    """
    clean_url = clean_input_url(options.url)
    parsed_base = urlparse(clean_url)
    
    try:
        async with httpx.AsyncClient(headers=DEFAULT_HEADERS, follow_redirects=True, timeout=15.0) as client:
            res = await client.get(clean_url)
            res.raise_for_status()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Map failed to fetch URL: {exc}")

    soup = BeautifulSoup(res.text, "html.parser")
    links = set()
    
    for a in soup.find_all("a", href=True):
        absolute = absolute_url(clean_url, a["href"])
        if absolute:
            parsed_link = urlparse(absolute)
            if options.include_subdomains:
                if parsed_link.hostname and parsed_base.hostname in parsed_link.hostname:
                    links.add(absolute)
            else:
                if parsed_link.hostname == parsed_base.hostname:
                    links.add(absolute)
                    
    return {
        "success": True, 
        "url": clean_url, 
        "links": list(links)[:options.limit]
    }


@app.post("/crawl")
async def crawl_url(options: CrawlOptions):
    """
    Discovers links via /map, then concurrently scrapes them up to the limit.
    """
    map_res = await map_url(MapOptions(url=options.url, limit=options.limit))
    target_urls = map_res["links"]
    
    if not target_urls:
        target_urls = [clean_input_url(options.url)]
        
    tasks = []
    scrape_opts = options.scrape_options or ScrapeOptions(url="")
    
    # Handle Pydantic v1 vs v2 dict conversion
    base_opts_dict = scrape_opts.model_dump() if hasattr(scrape_opts, "model_dump") else scrape_opts.dict()
    
    for u in target_urls:
        opts_dict = base_opts_dict.copy()
        opts_dict["url"] = u
        tasks.append(scrape_engine(**opts_dict))
        
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    successful = [r for r in results if isinstance(r, dict) and r.get("success")]
    errors = [str(r) for r in results if isinstance(r, Exception)]
    
    return {
        "success": True, 
        "crawled": len(successful), 
        "data": successful, 
        "errors": errors
    }


@app.post("/search")
async def search_web(options: SearchOptions):
    """
    Queries the web and returns top matching links and snippets.
    """
    if not options.query.strip():
        raise HTTPException(status_code=400, detail="A search query is required.")

    try:
        from duckduckgo_search import DDGS
    except ImportError:
        raise HTTPException(
            status_code=500, 
            detail="Package missing. Please run: pip install duckduckgo-search"
        )
        
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(options.query, max_results=options.limit))
            
        return {
            "success": True, 
            "query": options.query, 
            "results": results
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Search failed: {exc}")


@app.post("/agent")
async def clara_agent(options: AgentOptions):
    """
    Clara: Autonomous Web Agent for ReClaire. Searches, scrapes, and outputs JSON or CSV.
    """
    require_gemini()
    target_urls = options.urls or []
    
    # 1. Autonomous Search (If no URLs provided)
    if not target_urls:
        search_res = await search_web(SearchOptions(query=options.prompt, limit=3))
        target_urls = [r.get("href") for r in search_res.get("results", []) if r.get("href")]
        
    if not target_urls:
        raise HTTPException(status_code=400, detail="Clara could not find any relevant URLs to research.")
        
    # 2. Concurrent Scraping
    tasks = [scrape_engine(url=u, formats=["markdown"], only_main_content=True) for u in target_urls[:3]]
    scraped_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    valid_contexts = [
        f"Source: {res['url']}\n\n{res.get('markdown', '')}" 
        for res in scraped_results if isinstance(res, dict) and res.get("success")
    ]
            
    aggregated_context = "\n\n---\n\n".join(valid_contexts)
    
    # 3. CSV Output (Svc)
    if options.output_format.lower() == "csv":
        prompt = f"""
You are Clara, an autonomous research agent for ReClaire.
Extract the information requested by the user and format it STRICTLY as valid CSV.
Return ONLY the raw CSV text, including headers. Do not use markdown wrappers.

User Request: {options.prompt}
Gathered Web Sources:
{aggregated_context[:30000]}
"""
        response = gemini_client.models.generate_content(model="gemini-3.6-flash", contents=prompt)
        csv_text = response.text.strip()
        
        # Strip markdown formatting if Gemini includes it
        csv_text = re.sub(r"^```(?:csv)?\s*", "", csv_text, flags=re.IGNORECASE)
        csv_text = re.sub(r"\s*```$", "", csv_text)
        
        return {"agent": "Clara", "format": "csv", "data": csv_text}
        
    # 4. JSON Output
    else:
        schema = options.json_schema or {"result": "string"}
        prompt = f"""
You are Clara, an autonomous research agent for ReClaire.
Synthesize the gathered web content to fulfill the user's request.

User Request: {options.prompt}
Gathered Web Sources:
{aggregated_context[:30000]}

You MUST return valid JSON adhering strictly to this schema:
{json.dumps(schema, ensure_ascii=False)}
"""
        response = gemini_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        return {"agent": "Clara", "format": "json", "data": parse_json_response(response.text)}


# ============================================================
# API ENDPOINT
# ============================================================

@app.post("/scrape")
async def scrape_post(
    options: ScrapeOptions,
):

    try:

        # Pydantic v2
        if hasattr(options, "model_dump"):
            data = options.model_dump()

        # Pydantic v1
        else:
            data = options.dict()

        return await scrape_engine(
            **data
        )

    except HTTPException:
        raise

    except Exception as exc:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                "Unexpected server error: "
                f"{str(exc)}"
            ),
        )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
async def root():

    return {
        "name": "ReClaire API",
        "version": "2.1.0",
        "status": "online",
        "endpoint": "/scrape",
    }


@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "gemini_configured": gemini_client is not None,
    }


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                8000,
            )
        ),
    )
