
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import easyocr
import openai
import requests
import io
from PIL import Image

app = FastAPI()

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = "YOUR_OPENAI_API_KEY"
SERPER_API_KEY = "YOUR_SERPER_API_KEY"

reader = easyocr.Reader(['en'])

class FactCheckResult(BaseModel):
    extracted_text: str
    fact_check: str
    citations: list

@app.post("/factcheck", response_model=FactCheckResult)
async def factcheck(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    text_results = reader.readtext(image)
    extracted_text = " ".join([res[1] for res in text_results])

    # Search online for context (simplified)
    search_response = requests.get(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": SERPER_API_KEY},
        json={"q": extracted_text[:100]}
    ).json()

    links = [item['link'] for item in search_response.get("organic", [])[:3]]

    # Ask OpenAI to fact-check
    prompt = f"Fact-check the following claim(s) and cite sources:\n\n{extracted_text}\n\nSources: {links}"
    chat_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a fact-checking assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    fact_check = chat_response.choices[0].message.content.strip()

    return FactCheckResult(
        extracted_text=extracted_text,
        fact_check=fact_check,
        citations=links
    )
