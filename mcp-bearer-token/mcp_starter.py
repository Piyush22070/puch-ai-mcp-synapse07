import asyncio
from typing import Optional, Annotated
import os
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import TextContent, ImageContent, INVALID_PARAMS, INTERNAL_ERROR
from pydantic import BaseModel, Field, AnyUrl

import markdownify
import httpx
import readabilipy

# --- Load environment variables ---
load_dotenv()

TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY")


assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"
assert GEMINI_API_KEY is not None, "Please set GEMINI_API_KEY in your .env file"
assert SERPER_API_KEY is not None, "Please set SERPER_API_KEY in your .env file"

# --- Auth Provider ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="puch-client",
                scopes=["*"],
                expires_at=None,
            )
        return None

# --- Rich Tool Description model ---
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None

# --- Fetch Utility Class ---
class Fetch:
    USER_AGENT = "Puch/1.0 (Autonomous)"

    @classmethod
    async def fetch_url(
        cls,
        url: str,
        user_agent: str,
        force_raw: bool = False,
    ) -> tuple[str, str]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    follow_redirects=True,
                    headers={"User-Agent": user_agent},
                    timeout=30,
                )
            except httpx.HTTPError as e:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"))

            if response.status_code >= 400:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url} - status code {response.status_code}"))

            page_raw = response.text

        content_type = response.headers.get("content-type", "")
        is_page_html = "text/html" in content_type

        if is_page_html and not force_raw:
            return cls.extract_content_from_html(page_raw), ""

        return (
            page_raw,
            f"Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n",
        )

    @staticmethod
    def extract_content_from_html(html: str) -> str:
        """Extract and convert HTML content to Markdown format."""
        ret = readabilipy.simple_json.simple_json_from_html_string(html, use_readability=True)
        if not ret or not ret.get("content"):
            return "<error>Page failed to be simplified from HTML</error>"
        content = markdownify.markdownify(ret["content"], heading_style=markdownify.ATX)
        return content

    @staticmethod
    async def google_search_links(query: str, num_results: int = 5) -> list[str]:
        """
        Perform a scoped DuckDuckGo search and return a list of job posting URLs.
        (Using DuckDuckGo because Google blocks most programmatic scraping.)
        """
        ddg_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        links = []

        async with httpx.AsyncClient() as client:
            resp = await client.get(ddg_url, headers={"User-Agent": Fetch.USER_AGENT})
            if resp.status_code != 200:
                return ["<error>Failed to perform search.</error>"]

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", class_="result__a", href=True):
            href = a["href"]
            if "http" in href:
                links.append(href)
            if len(links) >= num_results:
                break

        return links or ["<error>No results found.</error>"]

# --- MCP Server Setup ---
mcp = FastMCP(
    "Job Finder MCP Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)

# --- Tool: validate (required by Puch) ---
@mcp.tool
async def validate() -> str:
    return MY_NUMBER


# --- Tool: find_product_online (FIXED WITH ERROR HANDLING) ---
# add the custom tool for product here

# --- Tool: find_similar_products ---
FIND_SIMILAR_PRODUCTS_DESCRIPTION = RichToolDescription(
    description="Analyze an image to identify products and find similar items online using Serper API",
    use_when="Use this when user sends an image of a product and wants to find similar items or shopping links",
    side_effects="Returns product information and shopping links from various online retailers",
)

@mcp.tool(description=FIND_SIMILAR_PRODUCTS_DESCRIPTION.model_dump_json())
async def find_similar_products(
    puch_image_data: Annotated[str, Field(description="Base64-encoded image data of the product")] = None,
    search_query: Annotated[str | None, Field(description="Optional text description to refine search")] = None,
    max_results: Annotated[int, Field(description="Maximum number of products to return")] = 5,
) -> str:
    """
    Analyze product image using Gemini Vision and search for similar products using Serper API
    """
    print("Api Hitted!")
    try:
        if not puch_image_data:
            raise McpError(ErrorData(code=INVALID_PARAMS, message="Image data is required"))

        # Step 1: Analyze image with Gemini Vision
        print("Passed Puch_image_data")
        product_description = await analyze_product_image(puch_image_data)
        
        print("Gemeni Respone "+product_description)
        
        # Step 2: Use search query if provided, otherwise extract keywords from Gemini description
        search_terms = search_query if search_query else extract_keywords_from_gemini(product_description)

        # Step 3: Search for products using Serper
        products = await search_products_serper(search_terms, max_results)
        
        # Step 4: Format response
        response = f"🔍 **Product Analysis Results**\n\n"
        response += f"**Detected Product:** {product_description}\n\n"
        
        if search_query:
            response += f"**Search Query Used:** {search_query}\n\n"
        
        response += "**Similar Products Found:**\n\n"
        
        for i, product in enumerate(products, 1):
            response += f"**{i}. {product['title']}**\n"
            response += f"   💰 Price: {product.get('price', 'N/A')}\n"
            response += f"   🏪 Store: {product.get('source', 'N/A')}\n"
            response += f"   🔗 Link: {product['link']}\n"
            if product.get('rating'):
                response += f"   ⭐ Rating: {product['rating']}\n"
            response += "\n"
        
        return response
        
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Product search failed: {str(e)}"))

import re
def extract_keywords_from_gemini(text: str) -> str:
    """
    Extract the 'Keywords for Online Shopping Searches' section from Gemini's response text,
    join them into a concise, comma-separated string suitable for search query.
    """
    match = re.search(
        r"Keywords for Online Shopping Searches:\n((?:\*.*\n)+)", text, re.MULTILINE
    )
    if match:
        keywords_block = match.group(1)
        keywords = [line.strip().lstrip("* ").strip() for line in keywords_block.strip().splitlines()]
        keywords_str = ", ".join(keywords)
        # Limit length to avoid API limits, e.g., max 200 characters
        return keywords_str[:200]
    else:
        # Fallback: Use first 200 chars of full text
        return text[:200]

async def analyze_product_image(image_data: str) -> str:
    """
    Use Gemini Vision API to analyze the product in the image.
    """

    try:
        print("Reached Gemini try")
        async with httpx.AsyncClient() as client:
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": (
                                    "Analyze this image briefly and concisely. Provide a 3-4 sentence summary describing the product type, main color, and key features. Use keywords useful for online shopping."
                                )
                            },
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": image_data
                                }
                            }
                        ]
                    }
                ]
            }

            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key={GEMINI_API_KEY}",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=20
            )

            print("Status Code of Gemini:", response.status_code)

            if response.status_code != 200:
                raise Exception(f"Gemini API error: {response.status_code} - {response.text}")

            result = response.json()
            print("Response from Gemini:", result)

            # Validate response structure
            candidates = result.get("candidates")
            if not candidates or len(candidates) == 0:
                raise Exception("No candidates found in Gemini response")

            content = candidates[0].get("content")
            if not content:
                raise Exception("No content found in Gemini response candidate")

            parts = content.get("parts")
            if not parts or len(parts) == 0:
                # If 'parts' missing, fallback to raw text or whole content
                fallback_text = content.get("text") or "general product search"
                return fallback_text.strip()

            return parts[0].get("text", "general product search").strip()

    except Exception as e:
        print("Error in Gemini:", e)
        return "general product search"

async def search_products_serper(query: str, max_results: int = 5) -> list:
    """
    Search for products using Serper Shopping API
    """
    try:
        async with httpx.AsyncClient() as client:
            payload = {
                "q": query,
                "gl": "in",  # Country code
                "hl": "en",  # Language
                "num": max_results
            }
            
            headers = {
                "X-API-KEY": SERPER_API_KEY,
                "Content-Type": "application/json"
            }
            
            response = await client.post(
                "https://google.serper.dev/shopping",
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"Serper API error: {response.status_code} - {response.text}")
            
            data = response.json()
            
            products = []
            shopping_results = data.get('shopping', [])
            
            for item in shopping_results[:max_results]:
                product = {
                    'title': item.get('title', 'Unknown Product'),
                    'price': item.get('price', 'N/A'),
                    'source': item.get('source', 'Unknown Store'),
                    'link': item.get('link', '#'),
                    'rating': item.get('rating'),
                    'reviews': item.get('reviews')
                }
                products.append(product)
            
            return products
            
    except Exception as e:
        raise Exception(f"Product search failed: {str(e)}")


# Don't forget to add this to your environment variables:
# SERPER_API_KEY = os.environ.get("SERPER_API_KEY")
# assert SERPER_API_KEY is not None, "Please set SERPER_API_KEY in your .env file"
# --- Run MCP Server ---
async def main():
    print("🚀 Starting MCP server on http://0.0.0.0:8086")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    asyncio.run(main())