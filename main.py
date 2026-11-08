import os
import json
import requests

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()

# Api
app = FastAPI(
    title="Vehicle Maintenance Analysis API",
    description="API for extracting vehicle queries, fetching workorder data, and serving LLM maintenance analysis.",
    version="1.0.0"
)

# llms
llm = ChatGroq(
    model= "llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=3000,
)

jsonLLM = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0,
    model_kwargs={"response_format": {"type": "json_object"}}
)

# Request the url for vehicleData

url = os.getenv("url")

def fetchVehicleData(vehicle_id, limit=10):
    if not vehicle_id:
        print("No vehicle_id provided.")
        return None

    api_key = os.getenv("DATA_API_KEY")
    
    params = {
        "vehicle_id": vehicle_id,
        "limit": limit,
        "sort": "desc"
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}" if api_key else ""
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"API Request failed: {e}")
        return None


prompt = PromptTemplate(
    input_variables=["query"],
    template="""
Extract the vehicle ID (code/number) and vehicle name (model/brand) from the following query.

Return ONLY a valid JSON object with these exact keys:
- "vehicle_id": integer or null if not found
- "vehicle_name": string or null if not found

Query: {query}
"""
)

chain = prompt | jsonLLM
query = "Can you share a deep analysis for car id 34?"

response = chain.invoke({"query": query})

data = json.loads(response.content)

vehicle_id = data.get("vehicle_id")
vehicle_name = data.get("vehicle_name")

# print(f"Vehicle : {vehicle_name}")
print(f"Vehicle id : {vehicle_id}")


rawVehicleData = fetchVehicleData(vehicle_id=vehicle_id)
# print(f"Raw vehicle data: {rawVehicleData}")


# Normalize the raw data 

def parseVehicleData(rawVehicleData):
    """
    Parses alternating workorder headers and line item details into clean dicts.
    """
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
                    "subtotal": item.get("subtotal")
                })
        
        parsedRecords.append({
            "workorder_no": header.get("workorder_no"),
            "repair_date": header.get("repair_date"),
            "odometer_km": header.get("odometer"),
            "distance_driven_since_last_service_km": header.get("diff_km"),
            "total_cost": header.get("total_amount"),
            "services_performed": tasks
        })

    return parsedRecords


vehicleData = parseVehicleData(rawVehicleData)

# print(f"Parsed Vehicle Data : {vehicleData}")


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
"""
)

parser = StrOutputParser()

vehicleChain = vechicleDataPrompt | llm | parser


vehicleSuggestion = vehicleChain.invoke(dict(
    query=query,
    vehicleData=json.dumps(vehicleData)
))
print(vehicleSuggestion)


class AnalysisRequest(BaseModel):
    query: str = Field(
        ..., 
        description="The user's maintenance inquiry containing a vehicle ID or name.",
        example="Can you analyze the maintenance history for vehicle ID 55?"
    )

class AnalysisResponse(BaseModel):
    vehicle_name: str | None
    vehicle_id: int | None
    parsed_vehicle_data: list
    analysis: str


@app.post("/analyze-vehicle", response_model=AnalysisResponse)
async def analyze_vehicle(request: AnalysisRequest):
    # Dynamically receives ANY input query sent by the user/client
    user_query = request.query

    # 1. Dynamically extract Vehicle ID and Name using the LLM chain
    response = chain.invoke({"query": user_query})
    
    # Handle both string and dict response formats safely
    if hasattr(response, "content"):
        data = json.loads(response.content)
    elif isinstance(response, dict):
        data = response
    else:
        data = json.loads(str(response))

    vehicle_id = data.get("vehicle_id")
    vehicle_name = data.get("vehicle_name")

    # If the LLM failed to extract an ID from the dynamic query
    if not vehicle_id:
        raise HTTPException(
            status_code=400,
            detail="Could not extract a valid vehicle ID from your query. Please specify a vehicle ID."
        )

    # 2. Fetch data dynamically for the extracted vehicle_id
    rawVehicleData = fetchVehicleData(vehicle_id=vehicle_id)
    
    if rawVehicleData is None:
        raise HTTPException(
            status_code=404, 
            detail=f"Failed to retrieve maintenance data for vehicle ID {vehicle_id}."
        )

    # 3. Parse Raw Data
    vehicleData = parseVehicleData(rawVehicleData)

    # 4. Generate LLM Analysis dynamically based on the specific query and data
    vehicleSuggestion = vehicleChain.invoke({
        "query": user_query,
        "vehicleData": json.dumps(vehicleData)
    })

    return AnalysisResponse(
        vehicle_name=vehicle_name,
        vehicle_id=vehicle_id,
        parsed_vehicle_data=vehicleData,
        analysis=vehicleSuggestion
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)