# ShopWise AI

ShopWise AI is an AI shopping research assistant built with Streamlit and Agno. It combines a natural-language shopping brief, web search, product comparisons, qualitative trade-offs, and a source list.

## Requirements

- Python 3.10 or later
- A Gemini API key, or an OpenAI API key if selecting OpenAI
- Internet access for model requests and web search

## Setup on Windows

Open PowerShell in the repository root and run:

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

In `.env`, set `GEMINI_API_KEY` to your Gemini key, save, and close Notepad. Keep this file private; `.env` is excluded by `.gitignore`. Never put a real key in `.env.example`, source code, or a commit.

Start the app:

```powershell
streamlit run app.py
```

Open the local URL Streamlit prints, usually http://localhost:8501. Stop it with **Ctrl+C** in PowerShell.

## Configuration

Gemini is the default provider:

- `AI_PROVIDER=gemini`
- `GEMINI_API_KEY` — required Gemini API key
- `GEMINI_MODEL` — optional model name; defaults to `gemini-3.8-flash`
- `GEMINI_FALLBACK_MODEL` — optional comma-separated fallback models; leave blank to use only `GEMINI_MODEL`

To use an OpenAI-compatible provider, set `AI_PROVIDER=openai` and configure `OPENAI_API_KEY`. `OPENAI_MODEL` and `OPENAI_BASE_URL` are optional.

After changing `.env`, restart Streamlit. The default configuration uses only `gemini-3.8-flash`. If you add fallbacks, use only model IDs currently available to your Gemini API key. If Gemini reports an exhausted account quota, the app reports it instead of presenting sample products as live research.

## How to use

Describe a product category, budget, intended user, and your priorities. Select **Research Products** to perform a live search and AI-assisted comparison. The brief, product cards, comparison table, qualitative decision matrix, trade-offs, shortlist, and verifiable source links appear below the request form.

Prices, availability, ratings, and specifications can change; confirm important details on the linked sources before buying.
