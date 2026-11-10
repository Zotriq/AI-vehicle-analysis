"""Vehicle maintenance analysis: plain-English question -> work orders -> maintenance report.

Data comes from the fleet API set in `url` (+ DATA_API_KEY). Without it, the bundled sample
fleet in sample_data/workorders.json is used, in the same raw format as the API.
"""

import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
SAMPLE_DATA = Path(__file__).parent / "sample_data" / "workorders.json"

app = FastAPI(
    title="Vehicle Maintenance Analysis API",
    description="API for extracting vehicle queries, fetching workorder data, and serving LLM maintenance analysis.",
    version="1.1.0",
)


def make_llms(model: str = MODEL):
    """(analysis LLM, JSON-mode extraction LLM). Created lazily so importing this module makes no API calls."""
    llm = ChatGroq(model=model, temperature=0.1, max_tokens=3000)
    json_llm = ChatGroq(model=model, temperature=0, model_kwargs={"response_format": {"type": "json_object"}})
    return llm, json_llm


def fetchVehicleData(vehicle_id, limit=10):
    """Latest work orders for a vehicle, in the fleet API's raw format (alternating header / line items)."""
    if not vehicle_id:
        return None
    url = os.getenv("url")
    if not url:
        fleet = json.loads(SAMPLE_DATA.read_text())
        record = fleet.get(str(vehicle_id))
        return record["workorders"][: limit * 2] if record else None
    api_key = os.getenv("DATA_API_KEY")
    params = {"vehicle_id": vehicle_id, "limit": limit, "sort": "desc"}
    headers = {"Authorization": f"Bearer {api_key}" if api_key else ""}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API Request failed: {e}")
        return None


def parseVehicleData(rawVehicleData):
    """Parses alternating workorder headers and line item details into clean dicts."""
    if not rawVehicleData or not isinstance(rawVehicleData, list):
        return []

    parsedRecords = []
    for i in range(0, len(rawVehicleData) - 1, 2):
        header_list = rawVehicleData[i]
        items_list = rawVehicleData[i + 1]
        if not header_list:
            continue
        header = header_list[0]

        tasks = []
        for item_wrapper in items_list:
            if item_wrapper and isinstance(item_wrapper, list) and len(item_wrapper) > 0:
                item = item_wrapper[0]
                tasks.append({
                    "service_task": item.get("service_task", "").strip(),
                    "labor_cost": item.get("labor"),
                    "parts_cost": item.get("parts"),
                    "subtotal": item.get("subtotal"),
                })

        parsedRecords.append({
            "workorder_no": header.get("workorder_no"),
            "repair_date": header.get("repair_date"),
            "odometer_km": header.get("odometer"),
            "distance_driven_since_last_service_km": header.get("diff_km"),
            "total_cost": header.get("total_amount"),
            "services_performed": tasks,
        })
    return parsedRecords


extract_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
Extract the vehicle ID (code/number) and vehicle name (model/brand) from the following query.

Return ONLY a valid JSON object with these exact keys:
- "vehicle_id": integer or null if not found
- "vehicle_name": string or null if not found

Query: {query}
""",
)

vechicleDataPrompt = PromptTemplate(
    input_variables=["query", "vehicleData"],
    template="""
You are an expert automotive service advisor and maintenance analyst. Your goal is to analyze the provided vehicle maintenance logs, address the user's specific query, and provide a clear, professional summary alongside actionable recommendations.

### Instructions:
1. **Analyze Records:** Review the raw vehicle data and extract all key events, repair trends, and recurring mechanical issues.
2. **Address Query:** Directly answer the user's query using the maintenance context provided.
3. **Chronological Summary:** Present key services performed in chronological order, highlighting replaced parts and major inspections.
4. **Actionable Recommendations:** Identify overdue services, recurring issues, or potential preventive steps based on the vehicle's history.

---

### Input data:
User Query: {query}
Vehicle Data: {vehicleData}

---
### Output format:
Provide your response strictly in the following structure:

1. **Direct Answer / Overview**: Brief response tailored specifically to the user's query.
2. **Maintenance Summary**:
   - Organized chronologically (Date | Service Performed | Key Replaced Parts / Findings).
   - Highlight any noticeable patterns (e.g., frequent brake wear, repeated fluid leaks).
3. **Expert Recommendations & Next Steps**:
   - Brief, clear advice on upcoming maintenance or preventive actions needed.
   - Any warning signs or items to keep an eye on.
""",
)


class VehicleNotFound(Exception):
    pass


def extract_vehicle(query: str, json_llm) -> dict:
    response = (extract_prompt | json_llm).invoke({"query": query})
    content = response.content if hasattr(response, "content") else response
    return json.loads(content) if isinstance(content, str) else dict(content)


def analyze(query: str, llm=None, json_llm=None) -> dict:
    """Full pipeline. Raises ValueError (no vehicle ID in the query) or VehicleNotFound."""
    if llm is None or json_llm is None:
        llm, json_llm = make_llms()
    data = extract_vehicle(query, json_llm)
    vehicle_id, vehicle_name = data.get("vehicle_id"), data.get("vehicle_name")
    if not vehicle_id:
        raise ValueError("Could not extract a valid vehicle ID from your query. Please specify a vehicle ID.")
    raw = fetchVehicleData(vehicle_id=vehicle_id)
    if raw is None:
        raise VehicleNotFound(f"Failed to retrieve maintenance data for vehicle ID {vehicle_id}.")
    vehicleData = parseVehicleData(raw)
    analysis = (vechicleDataPrompt | llm | StrOutputParser()).invoke(
        {"query": query, "vehicleData": json.dumps(vehicleData)})
    return {"vehicle_name": vehicle_name, "vehicle_id": vehicle_id, "parsed_vehicle_data": vehicleData, "analysis": analysis}


class AnalysisRequest(BaseModel):
    query: str = Field(
        ...,
        description="The user's maintenance inquiry containing a vehicle ID or name.",
        examples=["Can you analyze the maintenance history for vehicle ID 55?"],
    )


class AnalysisResponse(BaseModel):
    vehicle_name: str | None
    vehicle_id: int | None
    parsed_vehicle_data: list
    analysis: str


@app.post("/analyze-vehicle", response_model=AnalysisResponse)
async def analyze_vehicle(request: AnalysisRequest):
    try:
        return AnalysisResponse(**analyze(request.query))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except VehicleNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
