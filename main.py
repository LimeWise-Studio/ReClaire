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

    raw_url = raw_url.strip()

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

def extract_image_urls(
    soup: BeautifulSoup,
    base_url: str,
) -> List[str]:

    images = []

    # --------------------------------------------------------
    # Normal <img> elements
    # --------------------------------------------------------

    for img in soup.find_all("img"):

        attributes = [
            "src",
            "data-src",
            "data-lazy-src",
            "data-original",
            "data-image",
            "data-url",
        ]

        for attr in attributes:

            value = img.get(attr)

            url = absolute_url(
                base_url,
                value,
            )

            if url:
                images.append(url)

        # ----------------------------------------------------
        # srcset
        # ----------------------------------------------------

        srcset = img.get("srcset")

        if srcset:

            for candidate in srcset.split(","):

                candidate = candidate.strip()

                if not candidate:
                    continue

                # srcset format:
                # image.jpg 1x
                # image-large.jpg 1200w

                candidate_url = candidate.split()[0]

                url = absolute_url(
                    base_url,
                    candidate_url,
                )

                if url:
                    images.append(url)

    # --------------------------------------------------------
    # <source> inside <picture>
    # --------------------------------------------------------

    for source in soup.find_all("source"):

        srcset = source.get("srcset")

        if srcset:

            for candidate in srcset.split(","):

                candidate = candidate.strip()

                if not candidate:
                    continue

                candidate_url = candidate.split()[0]

                url = absolute_url(
                    base_url,
                    candidate_url,
                )

                if url:
                    images.append(url)

        src = source.get("src")

        url = absolute_url(
            base_url,
            src,
        )

        if url:
            images.append(url)

    return unique_list(images)


def extract_image_metadata(
    soup: BeautifulSoup,
    base_url: str,
) -> List[Dict[str, Any]]:

    results = []

    for img in soup.find_all("img"):

        candidate = None

        for attr in (
            "src",
            "data-src",
            "data-lazy-src",
            "data-original",
            "data-image",
            "data-url",
        ):
            value = img.get(attr)

            if value:
                candidate = absolute_url(
                    base_url,
                    value,
                )

                if candidate:
                    break

        if not candidate:
            srcset = img.get("srcset")

            if srcset:
                first = srcset.split(",")[0].strip()

                if first:
                    candidate = absolute_url(
                        base_url,
                        first.split()[0],
                    )

        if not candidate:
            continue

        results.append(
            {
                "url": candidate,
                "alt": safe_text(img.get("alt")),
                "title": safe_text(img.get("title")),
                "width": img.get("width"),
                "height": img.get("height"),
            }
        )

    # Deduplicate by URL.
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

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    if soup.title:
        assets["title"] = safe_text(
            soup.title.get_text()
        )

    # --------------------------------------------------------
    # Meta helper
    # --------------------------------------------------------

    def get_meta(
        *,
        name: Optional[str] = None,
        property_name: Optional[str] = None,
    ) -> Optional[str]:

        if name:

            tag = soup.find(
                "meta",
                attrs={"name": name},
            )

        else:

            tag = soup.find(
                "meta",
                attrs={"property": property_name},
            )

        if tag:
            return safe_text(
                tag.get("content")
            )

        return None

    # --------------------------------------------------------
    # Standard metadata
    # --------------------------------------------------------

    assets["description"] = (
        get_meta(name="description")
        or get_meta(property_name="og:description")
    )

    assets["site_name"] = (
        get_meta(property_name="og:site_name")
        or get_meta(name="application-name")
    )

    assets["og_image"] = absolute_url(
        base_url,
        get_meta(property_name="og:image"),
    )

    assets["hero_image"] = assets["og_image"]

    assets["twitter_image"] = absolute_url(
        base_url,
        get_meta(name="twitter:image"),
    )

    assets["theme_color"] = (
        get_meta(name="theme-color")
        or get_meta(name="msapplication-TileColor")
    )

    # --------------------------------------------------------
    # Favicon
    # --------------------------------------------------------

    icon_candidates = []

    for link in soup.find_all("link"):

        rel = link.get("rel", [])

        if isinstance(rel, str):
            rel = [rel]

        rel_lower = {
            str(item).lower()
            for item in rel
        }

        href = link.get("href")

        if not href:
            continue

        if (
            "icon" in rel_lower
            or "shortcut icon" in rel_lower
            or "apple-touch-icon" in rel_lower
        ):

            icon_url = absolute_url(
                base_url,
                href,
            )

            if not icon_url:
                continue

            if "apple-touch-icon" in rel_lower:
                assets["apple_touch_icon"] = icon_url

            icon_candidates.append(
                {
                    "url": icon_url,
                    "rel": list(rel_lower),
                    "sizes": link.get("sizes"),
                    "type": link.get("type"),
                }
            )

    if icon_candidates:

        assets["favicon"] = (
            assets["apple_touch_icon"]
            or icon_candidates[0]["url"]
        )

    if not assets["favicon"]:

        assets["favicon"] = (
            f"https://www.google.com/s2/favicons"
            f"?domain={domain}&sz=64"
        )

    # --------------------------------------------------------
    # Web manifest
    # --------------------------------------------------------

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

        assets["manifest"] = absolute_url(
            base_url,
            manifest_link.get("href"),
        )

    # --------------------------------------------------------
    # Logo candidates
    # --------------------------------------------------------

    logo_candidates = []

    # Images whose alt/class/id suggests a logo.
    for img in soup.find_all("img"):

        combined = " ".join(
            [
                safe_text(img.get("alt")),
                safe_text(img.get("class")),
                safe_text(img.get("id")),
            ]
        ).lower()

        if "logo" not in combined:
            continue

        url = None

        for attr in (
            "src",
            "data-src",
            "data-lazy-src",
            "data-original",
        ):

            value = img.get(attr)

            if value:
                url = absolute_url(
                    base_url,
                    value,
                )

                if url:
                    break

        if url:
            logo_candidates.append(url)

    # SVGs with logo-like identifiers.
    for svg in soup.find_all("svg"):

        combined = " ".join(
            [
                safe_text(svg.get("id")),
                safe_text(svg.get("class")),
                safe_text(svg.get("aria-label")),
            ]
        ).lower()

        if "logo" in combined:
            logo_candidates.append(
                "inline-svg"
            )

    assets["logo_candidates"] = unique_list(
        logo_candidates
    )

    if assets["logo_candidates"]:
        assets["logo"] = assets["logo_candidates"][0]

    # --------------------------------------------------------
    # Font information
    # --------------------------------------------------------

    font_values = []

    for style in soup.find_all("style"):

        css = style.get_text(" ", strip=True)

        if not css:
            continue

        matches = re.findall(
            r"font-family\s*:\s*([^;}{]+)",
            css,
            flags=re.IGNORECASE,
        )

        for match in matches:

            cleaned = safe_text(match)

            if cleaned:
                font_values.append(cleaned)

    assets["fonts"] = unique_list(font_values)[:20]

    # --------------------------------------------------------
    # CSS variables
    # --------------------------------------------------------

    css_variables = {}

    for style in soup.find_all("style"):

        css = style.get_text(" ", strip=True)

        if not css:
            continue

        matches = re.findall(
            r"--([a-zA-Z0-9_-]+)\s*:\s*([^;}{]+)",
            css,
        )

        for key, value in matches:

            key = safe_text(key)
            value = safe_text(value)

            if key and value:
                css_variables[f"--{key}"] = value

    assets["css_variables"] = dict(
        list(css_variables.items())[:50]
    )

    # --------------------------------------------------------
    # Important metadata
    # --------------------------------------------------------

    meta_keys = [
        "author",
        "keywords",
        "application-name",
        "theme-color",
        "generator",
        "viewport",
    ]

    for key in meta_keys:

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
                    "The server returned no usable HTML."
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
        "version": "2.0.0",
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
