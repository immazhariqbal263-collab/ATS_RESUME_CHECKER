# 📄 ATS Resume Checker

A Streamlit app that scores a resume for ATS (Applicant Tracking System) compatibility and
suggests concrete improvements, powered by Google's Gemini Flash model.

## Features
- Upload a resume as **PDF, DOCX or TXT**
- Optional **job description** input for tailored keyword and relevance scoring
- Overall **ATS score (0-100)** plus a breakdown: formatting, keywords, content quality, structure, relevance
- Prioritised **improvement suggestions** (High / Medium / Low) with example fixes
- Found vs. missing **keywords**, strengths and formatting issues
- Downloadable **Markdown report**

> The score is an AI-generated estimate. Real ATS software varies, so treat it as guidance.

## Project structure
```
app.py            # Streamlit app
requirements.txt  # Python dependencies
README.md
```

## Run locally
1. Install Python 3.9+.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).
4. Provide the key in one of three ways:
   - Paste it into the app's sidebar, or
   - Set an environment variable: `export GEMINI_API_KEY="your-key"` (Windows: `set GEMINI_API_KEY=your-key`), or
   - Create `.streamlit/secrets.toml`:
     ```toml
     GEMINI_API_KEY = "your-key"
     ```
5. Start the app:
   ```bash
   streamlit run app.py
   ```

## Configuration
| Setting | Where | Default |
|---|---|---|
| `GEMINI_API_KEY` | secrets / env var / sidebar | none (required) |
| `GEMINI_MODEL` | secrets / env var / sidebar | `gemini-3.5-flash` |

If you get a "model not found" error, check the current Flash model names in
[Google AI Studio](https://aistudio.google.com) and enter one in the sidebar.

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app**, choose your repo, branch `main`, and main file `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
5. Click **Deploy**.

## Security notes
- Never commit your API key. Keep `.streamlit/secrets.toml` out of Git (add it to `.gitignore`).
- Uploaded resumes are processed in memory and sent to the Gemini API for analysis.
  Don't upload documents you aren't comfortable sharing with Google's API.

## Limitations
- Scanned or image-only PDFs have no extractable text; export a text-based PDF instead.
- Very long resumes are truncated to the first ~30,000 characters.
