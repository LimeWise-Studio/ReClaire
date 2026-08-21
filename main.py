import re
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as md

app = FastAPI(
    title="ReClaire API",
    description="Convert web pages into clean Markdown for AI prompts.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {
        "status": "online",
        "service": "ReClaire API",
        "usage": "Send GET request to /scrape?url=HTTPS_TARGET_URL"
    }

@app.get("/scrape")
async def scrape_to_markdown(url: str = Query(..., description="Target webpage URL")):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    # Extract ONLY the clean URL pattern using Regex (Ignores all surrounding brackets/markdown)
    url_match = re.search(r'https?://[^\s\]\)\>"]+', url)
    
    if not url_match:
        raise HTTPException(status_code=400, detail="Invalid URL format provided.")
        
    clean_url = url_match.group(0)

    # Fetch webpage raw HTML
    async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
        try:
            response = await client.get(clean_url, headers=headers)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail=f"Target URL returned status code {response.status_code}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch URL: {str(e)}")

    # Parse HTML and strip away non-content elements
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe", "noscript"]):
        tag.decompose()

    # Extract title and main content
    title = soup.title.string.strip() if soup.title and soup.title.string else "No Title Found"
    main_content = soup.find("main") or soup.find("article") or soup.body
    raw_html = str(main_content) if main_content else str(soup)

    # Convert cleaned HTML to Markdown
    clean_markdown = md(raw_html, heading_style="ATX", strip=["img"])

    return {
        "success": True,
        "url": clean_url,
        "title": title,
        "markdown": clean_markdown.strip()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
