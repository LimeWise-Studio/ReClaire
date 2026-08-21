from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as md

app = FastAPI(
    title="MarkdownScrape API",
    description="Convert web pages into clean Markdown for AI prompts.",
    version="1.0.0"
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {
        "status": "online",
        "usage": "Send GET request to /scrape?url=HTTPS_TARGET_URL"
    }

@app.get("/scrape")
async def scrape_to_markdown(url: str = Query(..., description="Target webpage URL")):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    # 1. Fetch webpage raw HTML
    async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
        try:
            response = await client.get(url, headers=headers)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail=f"Target URL returned status code {response.status_code}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch URL: {str(e)}")

    # 2. Parse HTML and strip away non-content elements
    soup = BeautifulSoup(response.text, "html.parser")
    
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe", "noscript"]):
        tag.decompose()

    # 3. Extract title and main content
    title = soup.title.string.strip() if soup.title and soup.title.string else "No Title Found"
    
    # Target article/main tag if available to minimize boilerplate
    main_content = soup.find("main") or soup.find("article") or soup.body
    raw_html = str(main_content) if main_content else str(soup)

    # 4. Convert cleaned HTML to Markdown
    clean_markdown = md(raw_html, heading_style="ATX", strip=["img"])

    return {
        "success": True,
        "url": url,
        "title": title,
        "markdown": clean_markdown.strip()
    }

if __name__ == "__main__":
    import uvicorn
    # Runs the server locally on localhost:8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
