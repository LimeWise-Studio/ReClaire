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

def clean_input_url(raw_url: str) -> str:
    """Bulletproof URL extractor that fixes broken Markdown links."""
    raw_url = raw_url.strip()
    
    # 1. If it's a mangled markdown link: [text](url)
    if "](" in raw_url:
        parts = raw_url.split("](", 1)
        url_part = parts[1]
        # Remove trailing markdown parenthesis
        if url_part.endswith(")"):
            url_part = url_part[:-1]
        return url_part.strip()
        
    # 2. If it's enclosed in brackets: [url]
    if raw_url.startswith("[") and raw_url.endswith("]"):
        return raw_url[1:-1].strip()
        
    # 3. Fallback: ensure http:// is present
    if not raw_url.startswith("http"):
        return "https://" + raw_url
        
    return raw_url

@app.get("/")
def home():
    return {"status": "online"}

@app.get("/scrape")
async def scrape_to_markdown(url: str = Query(..., description="Target webpage URL")):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    clean_url = clean_input_url(url)

    # 1. Fetch webpage (Try block ONLY for the network request)
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            response = await client.get(clean_url, headers=headers)
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Network error while connecting to the URL: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal system error: {str(e)}")

    # 2. Check status code OUTSIDE the try block! 
    # This prevents the server from catching its own 400 errors.
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail=f"The website returned an error (Status {response.status_code}).")

    # 3. Parse HTML securely
    try:
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe", "noscript"]):
            tag.decompose()

        # 4. Extract Markdown
        title = soup.title.string.strip() if soup.title and soup.title.string else "No Title Found"
        main_content = soup.find("main") or soup.find("article") or soup.body
        raw_html = str(main_content) if main_content else str(soup)
        clean_markdown = md(raw_html, heading_style="ATX", strip=["img"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing webpage content: {str(e)}")

    return {
        "success": True,
        "url": clean_url,
        "title": title,
        "markdown": clean_markdown.strip()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
