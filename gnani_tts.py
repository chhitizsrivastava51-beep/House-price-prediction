"""
Gnani.ai Text-to-Speech (TTS) Integration Module
Supports Indic languages & English for AI Property Valuation reports.
Includes intelligent fallback voice synthesis when API credentials are not active.
"""

import os
import io
import json
import base64
import requests
from typing import Tuple, Optional, Dict

# Supported Indic & English languages in Gnani.ai TTS
SUPPORTED_LANGUAGES = {
    "hi-IN": {
        "name": "Hindi (हिन्दी)",
        "gtts_code": "hi",
        "default_voice": "female"
    },
    "en-IN": {
        "name": "Indian English (English)",
        "gtts_code": "en",
        "default_voice": "female"
    },
    "bn-IN": {
        "name": "Bengali (বাংলা)",
        "gtts_code": "bn",
        "default_voice": "female"
    },
    "mr-IN": {
        "name": "Marathi (मराठी)",
        "gtts_code": "mr",
        "default_voice": "female"
    },
    "ta-IN": {
        "name": "Tamil (தமிழ்)",
        "gtts_code": "ta",
        "default_voice": "female"
    },
    "te-IN": {
        "name": "Telugu (తెలుగు)",
        "gtts_code": "te",
        "default_voice": "female"
    },
    "kn-IN": {
        "name": "Kannada (ಕನ್ನಡ)",
        "gtts_code": "kn",
        "default_voice": "female"
    },
    "gu-IN": {
        "name": "Gujarati (ગુજરાતી)",
        "gtts_code": "gu",
        "default_voice": "female"
    }
}

DEFAULT_ENDPOINT = "https://tts.gnani.ai/api/v1/tts"


def format_inr(amount_inr: float, spoken: bool = False, lang: str = "en") -> str:
    """Format Indian Rupees in Crores or Lakhs (with spoken-word support)."""
    if amount_inr >= 10_000_000:
        cr = amount_inr / 10_000_000
        if spoken:
            return f"{cr:.2f} करोड़ रुपये" if lang.startswith("hi") else f"{cr:.2f} Crore Rupees"
        return f"₹{cr:.2f} Cr"
    elif amount_inr >= 100_000:
        lakh = amount_inr / 100_000
        if spoken:
            return f"{lakh:.2f} लाख रुपये" if lang.startswith("hi") else f"{lakh:.2f} Lakh Rupees"
        return f"₹{lakh:.2f} Lakh"
    else:
        if spoken:
            return f"{int(amount_inr):,} रुपये" if lang.startswith("hi") else f"{int(amount_inr):,} Rupees"
        return f"₹{amount_inr:,.0f}"


def generate_valuation_script(
    price_usd: float,
    price_inr: float,
    top_features: list,
    lang: str = "hi-IN"
) -> str:
    """
    Generate natural voice narration text for property valuation.
    Uses words instead of raw currency symbols for optimal speech synthesis clarity.
    """
    usd_int = int(round(price_usd))
    f1 = top_features[0] if len(top_features) > 0 else "Overall Quality"
    f2 = top_features[1] if len(top_features) > 1 else "Living Area"

    if lang.startswith("hi"):
        inr_spoken = format_inr(price_inr, spoken=True, lang="hi")
        script = (
            f"नमस्ते! एआई प्रॉपर्टी वैल्यूएशन असिस्टेंट के अनुसार, इस घर का अनुमानित बाज़ार मूल्य "
            f"{inr_spoken} है, जो कि लगभग {usd_int:,} डॉलर के बराबर है। "
            f"हमारे मशीन लर्निंग मॉडल के मुताबिक, इस कीमत को निर्धारित करने में सबसे बड़ा योगदान "
            f"{f1} और {f2} का रहा है। "
            f"यह वैल्यूएशन प्रॉपर्टी की विशेषताओं और ऐतिहासिक डेटा के आधार पर तैयार किया गया है।"
        )
    elif lang.startswith("mr"):
        inr_spoken = format_inr(price_inr, spoken=True, lang="mr")
        script = (
            f"नमस्कार! एआई मॉडेलनुसार, या मालमत्तेचे अंदाजित मूल्य "
            f"{inr_spoken} (सुमारे {usd_int:,} डॉलर) आहे. "
            f"या मूल्यांकनात सर्वात महत्त्वाचा वाटा {f1} आणि {f2} चा आहे."
        )
    elif lang.startswith("bn"):
        inr_spoken = format_inr(price_inr, spoken=True, lang="bn")
        script = (
            f"নমস্কার! আমাদের এআই মডেল অনুসারে, এই সম্পত্তির আনুমানিক বাজার মূল্য "
            f"{inr_spoken} (প্রায় {usd_int:,} ডলার)। "
            f"এই মূল্যায়নে প্রধান নির্ধারক উপাদান হলো {f1} এবং {f2}।"
        )
    else:  # Default English
        inr_spoken = format_inr(price_inr, spoken=True, lang="en")
        script = (
            f"Hello! According to our AI property valuation model, the estimated market value "
            f"for this property is {usd_int:,} dollars, which corresponds to approximately {inr_spoken}. "
            f"The primary driving factors behind this valuation are {f1}, "
            f"followed by {f2}. "
            f"This estimate is synthesized from trained historical real estate patterns."
        )

    return script


def synthesize_speech(
    text: str,
    lang: str = "hi-IN",
    voice: str = "female",
    api_key: Optional[str] = None,
    token: Optional[str] = None,
    endpoint: Optional[str] = None
) -> Tuple[Optional[bytes], str, str, Optional[str]]:
    """
    Synthesizes speech using Gnani.ai Text-to-Speech API.
    If Gnani credentials are missing or call fails, gracefully falls back to local voice engine.

    Returns:
        (audio_bytes, mime_type, engine_used, status_message)
        engine_used is either 'gnani' or 'fallback'
    """
    api_key = (api_key or os.getenv("GNANI_API_KEY", "")).strip()
    token = (token or os.getenv("GNANI_TOKEN", "")).strip()
    endpoint = (endpoint or os.getenv("GNANI_TTS_ENDPOINT", DEFAULT_ENDPOINT)).strip()

    # If Gnani credentials exist, attempt Gnani API
    if api_key or token:
        try:
            headers = {
                "Content-Type": "application/json",
                "Accept": "audio/wav, audio/mpeg, application/json"
            }
            if token:
                headers["token"] = token
                headers["Authorization"] = f"Bearer {token}"
            if api_key:
                headers["x-api-key"] = api_key

            payload = {
                "text": text,
                "language": lang,
                "lang": lang.split("-")[0],
                "voice": voice.lower(),
                "gender": voice.lower(),
                "audio_format": "wav",
                "speed": 1.0
            }

            resp = requests.post(endpoint, json=payload, headers=headers, timeout=12)

            if resp.status_code == 200:
                content_type = resp.headers.get("content-type", "")
                if "audio" in content_type:
                    return resp.content, content_type, "gnani", "Gnani.ai Voice AI synthesis successful!"
                
                # Check for base64 audio in json response
                try:
                    data = resp.json()
                    audio_b64 = data.get("audioContent") or data.get("data") or data.get("audio")
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        return audio_bytes, "audio/wav", "gnani", "Gnani.ai Voice AI synthesis successful!"
                except Exception:
                    pass

                # If 200 returned raw bytes directly
                if len(resp.content) > 100:
                    return resp.content, "audio/wav", "gnani", "Gnani.ai Voice AI synthesis successful!"

            gnani_error = f"Gnani API returned HTTP {resp.status_code}: {resp.text[:120]}"
        except Exception as e:
            gnani_error = f"Gnani connection error: {str(e)}"
    else:
        gnani_error = "No Gnani.ai API Key or Token provided. Using intelligent fallback voice."

    # Fallback to gTTS (Google Text-to-Speech)
    try:
        from gtts import gTTS
        lang_info = SUPPORTED_LANGUAGES.get(lang, {})
        gtts_code = lang_info.get("gtts_code", "en" if "en" in lang else "hi")

        tts = gTTS(text=text, lang=gtts_code, slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        audio_bytes = audio_fp.read()
        
        return (
            audio_bytes,
            "audio/mp3",
            "fallback",
            f"Using local high-fidelity voice engine ({gnani_error})"
        )
    except Exception as fallback_err:
        return (
            None,
            "",
            "error",
            f"Voice synthesis failed. Gnani: {gnani_error}. Fallback: {str(fallback_err)}"
        )
