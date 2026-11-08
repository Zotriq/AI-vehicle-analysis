# AI Vehicle Maintenance Analysis

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070B-F55036?style=flat-square)

An API that turns raw vehicle work-order history into a clear maintenance report. Ask in plain English,
for example *"Can you share a deep analysis for car id 34?"*, and it returns a summary of past repairs,
recurring issues and recommended next steps.

## How it works

```
User question
   └─► LLM (JSON mode) extracts vehicle_id / vehicle_name
         └─► Fetch the latest work orders from the fleet data API
               └─► Normalize header + line-item records into clean JSON
                     └─► LLM acting as a service advisor writes the analysis
                           └─► JSON response
```

1. **Query understanding:** Llama 3.3 70B (via Groq) in JSON mode pulls the vehicle ID and name out of free text.
2. **Data retrieval:** the vehicle's most recent work orders are fetched from an external API (bearer-token auth).
3. **Normalization:** alternating work-order headers and line items are parsed into structured records.
4. **Analysis:** a second prompt reviews the history, spots repair trends and recurring faults, and gives recommendations.

## API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/analyze-vehicle` | Takes a natural-language query and returns the vehicle and its maintenance analysis |

Interactive docs are available at `/docs` once the server is running.

## Getting started

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_key
url=https://your-fleet-api/workorders
DATA_API_KEY=your_data_api_key
```

Run the API:

```bash
uvicorn main:app --reload
```

`app.ipynb` contains the same pipeline step by step, for experimenting.

## Tech stack

FastAPI · Pydantic · LangChain · Groq (Llama 3.3 70B) · Requests · python-dotenv
