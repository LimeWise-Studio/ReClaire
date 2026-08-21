import os
import json
import requests
from fastapi import FastAPI, Request, HTTPException
from supabase import create_client, Client

app = FastAPI(title="Pricely AI Agent Backend")

# Environment Variables
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SHOPIFY_STORE_DOMAIN = os.environ.get("SHOPIFY_STORE_DOMAIN")
SHOPIFY_ACCESS_TOKEN = os.environ.get("SHOPIFY_ACCESS_TOKEN")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

def call_gemini_api(prompt: str) -> dict:
    """Calls Gemini using direct REST API to eliminate SDK 404/v1beta version mismatches."""
    if not GEMINI_API_KEY:
        raise Exception("GEMINI_API_KEY is missing from environment variables.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"response_mime_type": "application/json"}
    }

    response = requests.post(url, headers=headers, json=payload, timeout=30)
    
    if response.status_code != 200:
        raise Exception(f"Gemini API returned HTTP {response.status_code}: {response.text}")

    res_json = response.json()
    raw_text = res_json['candidates'][0]['content']['parts'][0]['text']
    return json.loads(raw_text)


def update_shopify_price(variant_id: str, new_price: float):
    """Executes a price mutation on Shopify using GraphQL."""
    if not SHOPIFY_STORE_DOMAIN or not SHOPIFY_ACCESS_TOKEN:
        print("Shopify credentials not configured. Skipping live price mutation.")
        return False

    formatted_gid = variant_id if variant_id.startswith("gid://") else f"gid://shopify/ProductVariant/{variant_id}"
    url = f"https://{SHOPIFY_STORE_DOMAIN}/admin/api/2026-01/graphql.json"
    headers = {
        "Content-Type": "application/json",
        "X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN
    }

    mutation = """
    mutation productVariantUpdate($input: ProductVariantInput!) {
      productVariantUpdate(input: $input) {
        productVariant {
          id
          price
        }
        userErrors {
          field
          message
        }
      }
    }
    """

    variables = {
        "input": {
            "id": formatted_gid,
            "price": f"{new_price:.2f}"
        }
    }

    response = requests.post(url, json={"query": mutation, "variables": variables}, headers=headers)
    res_data = response.json()

    user_errors = res_data.get("data", {}).get("productVariantUpdate", {}).get("userErrors", [])
    if user_errors:
        print(f"Shopify Mutation Error: {user_errors}")
        return False

    print(f"Successfully updated Shopify price for {formatted_gid} to ${new_price:.2f}")
    return True


@app.get("/")
def read_root():
    return {"status": "Pricely Backend Engine Running"}


@app.post("/webhooks/shopify")
async def shopify_webhook(request: Request):
    payload = await request.json()
    
    product_id = str(payload.get("id") or payload.get("product_id", ""))
    title = payload.get("title", "Sample Product")
    variants = payload.get("variants", [{}])
    variant = variants[0] if variants else {}
    variant_id = str(variant.get("id", product_id))
    current_price = float(variant.get("price", 0.0) or 10.0)
    cost_price = float(variant.get("cost", 0.0) or (current_price * 0.5))
    inventory_quantity = int(variant.get("inventory_quantity", 15))

    if not product_id:
        raise HTTPException(status_code=400, detail="Invalid product ID")

    min_margin = 0.20
    max_increase = 0.15

    prompt = f"""
    You are an e-commerce dynamic pricing AI.
    Product: {title}
    Current Price: ${current_price}
    Unit Cost: ${cost_price}
    Inventory Left: {inventory_quantity}

    Rules:
    - Minimum Profit Margin: {min_margin * 100}%
    - Max Single Price Increase: {max_increase * 100}%

    Respond STRICTLY in raw JSON format with these exact keys:
    {{
        "should_change_price": true,
        "proposed_price": 88.0,
        "reasoning": "Explanation here"
    }}
    """

    try:
        rec = call_gemini_api(prompt)
    except Exception as e:
        print(f"CRITICAL GEMINI ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Gemini Execution Error: {str(e)}")

    proposed_price = float(rec.get("proposed_price", current_price))
    should_change = bool(rec.get("should_change_price", False))
    reasoning = str(rec.get("reasoning", ""))

    # Guardrail checks
    min_allowable = round(cost_price * (1 + min_margin), 2)
    max_allowable = round(current_price * (1 + max_increase), 2)

    if proposed_price < min_allowable:
        proposed_price = min_allowable
    elif proposed_price > max_allowable:
        proposed_price = max_allowable

    mutation_executed = False

    if should_change and proposed_price != current_price:
        mutation_executed = update_shopify_price(variant_id, proposed_price)

    if supabase and should_change:
        try:
            supabase.table("pricing_logs").insert({
                "product_id": product_id,
                "product_title": title,
                "old_price": current_price,
                "proposed_price": proposed_price,
                "ai_reasoning": reasoning,
                "status": "applied" if mutation_executed else "pending"
            }).execute()
        except Exception as log_err:
            print(f"Supabase Log Warning: {log_err}")

    return {
        "status": "success",
        "title": title,
        "current_price": current_price,
        "recommended_price": proposed_price,
        "should_change": should_change,
        "price_updated_in_shopify": mutation_executed,
        "ai_reasoning": reasoning
    }
