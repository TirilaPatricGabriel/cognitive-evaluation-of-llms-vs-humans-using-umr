# Cognitive Evaluation of LLMs vs Humans using UMR

## Setup

```bash
pip install fastapi uvicorn pandas pyyaml google-generativeai pydantic-settings
```

Create `.env` file:
```
GEMINI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

## Run

```bash
uvicorn app.main:app --reload
```

Go to `http://localhost:8000/docs`

## Endpoints

- `GET /load-data` - Load MultiplEYE Romanian and English texts
- `POST /reverse-engineer` - Generate prompts from 24 human texts
- `POST /generate-texts` - Generate AI texts from reversed prompts
