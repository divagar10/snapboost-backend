from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
import os

# ✅ Load API key securely from .env
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ✅ Shared Gemini model (to avoid cold start delays)
gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")

# ✅ Initialize FastAPI
app = FastAPI()

# ✅ CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://snapboost-frontend-2.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- Health Check ----------------------
@app.get("/")
async def root():
    return {"message": "✅ SnapBoost AI backend is running!"}

# ---------------------- 1. Caption / Idea / Launch Plan ----------------------
class CaptionRequest(BaseModel):
    event: str
    choice: str

@app.post("/generate-caption/")
async def generate_caption(request: CaptionRequest):
    choice = request.choice.strip().lower()
    if choice == "caption":
        prompt = f"Write a short, catchy social media caption for the launch of: {request.event}"
    elif choice == "idea":
        prompt = f"Give me creative marketing ideas for the launch of: {request.event}"
    elif choice == "launch plan":
        prompt = f"Create a simple and effective launch plan for: {request.event}"
    else:
        prompt = f"Tell me something helpful related to: {request.event}"
    try:
        response = gemini_model.generate_content(prompt)
        return {"caption": response.text.strip()}
    except Exception as e:
        return {"error": f"❌ Failed to generate caption. Reason: {str(e)}"}

# ---------------------- 2. Hashtag Generator ----------------------
class HashtagRequest(BaseModel):
    caption: str

@app.post("/generate-hashtags/")
async def generate_hashtags(data: HashtagRequest):
    prompt = f"""
    Generate a list of high-engagement social media hashtags based on the following caption. 
    Avoid spaces and return only the hashtags in Python list format.
    Caption: {data.caption}
    """
    try:
        response = gemini_model.generate_content(prompt)
        return {"hashtags": response.text.strip()}
    except Exception as e:
        return {"error": str(e)}

# ---------------------- 3. Idea Generator ----------------------
class IdeaRequest(BaseModel):
    platform: str
    niche: str
    audience_type: str

@app.post("/generate-ideas/")
async def generate_ideas(data: IdeaRequest):
    prompt = f"""
    You are a creative AI assistant for content creators. Generate 10 unique and viral content ideas for {data.platform} creators 
    in the "{data.niche}" niche. Each idea should be engaging, suitable for the {data.audience_type}, and relevant to current trends.
    Output the ideas as a numbered list.
    """
    try:
        response = gemini_model.generate_content(prompt)
        return {"ideas": response.text.strip()}
    except Exception as e:
        return {"error": f"❌ Failed to generate ideas. Reason: {str(e)}"}

# ---------------------- 4. Script Generator ----------------------
class ScriptInput(BaseModel):
    video_type: str
    topic: str

@app.post("/generate-script/")
async def generate_script(data: ScriptInput):
    video_type = data.video_type.strip().lower()
    topic = data.topic.strip()

    if "short" in video_type or "reel" in video_type:
        prompt = f"""
        Generate an engaging script for a 1-minute {video_type} about "{topic}".
        The script should have an attention-grabbing hook in the first few seconds,
        use a casual and high-energy tone, include visual/action suggestions,
        and end with a strong call to action.
        """
    else:
        prompt = f"""
        Generate a detailed script for a YouTube video about "{topic}".
        The script should last about 10 minutes and include:
        - Hook in the intro
        - Structured explanation in sections
        - Engaging storytelling tone
        - Visual suggestions for each section
        - Call to action at the end
        Format the response as a script with timestamps.
        """

    try:
        response = gemini_model.generate_content(prompt)
        return {"script": response.text.strip()}
    except Exception as e:
        return {"error": "❌ Failed to generate script. Please try again later."}

# ---------------------- 5. Thumbnail Prompt Generator ----------------------
class ThumbnailPromptRequest(BaseModel):
    topic: str
    style: str
    platform: str  # YouTube, Instagram

@app.post("/generate-thumbnail-prompt/")
async def generate_thumbnail_prompt(data: ThumbnailPromptRequest):
    prompt = f"""Generate a detailed and visually rich prompt for an AI image generator to create a thumbnail for a {data.platform} video.
Topic: "{data.topic}"
Style: {data.style}
Focus on visual details like colors, elements, facial expressions, mood, typography, and composition.
Avoid text in the prompt itself unless required."""
    try:
        response = gemini_model.generate_content(prompt)
        return {"prompt": response.text.strip()}
    except Exception as e:
        return {"error": str(e)}
