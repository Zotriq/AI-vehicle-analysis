import json

import pytest

import main


def test_parse_pairs_headers_with_line_items():
    raw = main.fetchVehicleData(34)
    records = main.parseVehicleData(raw)
    assert len(records) == 6
    first = records[0]
    assert first["workorder_no"] == "WO-34-118"
    assert first["odometer_km"] == 98450
    assert [t["service_task"] for t in first["services_performed"]][1] == "Front brake pads replaced"
    assert first["total_cost"] == sum(t["subtotal"] for t in first["services_performed"])


def test_unknown_vehicle_and_bad_input():
    assert main.fetchVehicleData(999) is None
    assert main.fetchVehicleData(None) is None
    assert main.parseVehicleData(None) == []
    assert main.parseVehicleData("not a list") == []


def test_limit_counts_work_orders_not_list_items():
    assert len(main.parseVehicleData(main.fetchVehicleData(34, limit=2))) == 2


class FakeLLM:
    """Stands in for ChatGroq in a LangChain pipe."""

    def __init__(self, reply):
        self.reply = reply

    def invoke(self, prompt, *a, **k):
        from langchain_core.messages import AIMessage
        return AIMessage(content=self.reply)

    def __call__(self, prompt):
        return self.invoke(prompt)


def test_analyze_end_to_end_with_fake_llms():
    from langchain_core.runnables import RunnableLambda

    json_llm = RunnableLambda(lambda p: FakeLLM(json.dumps({"vehicle_id": 55, "vehicle_name": None})).invoke(p))
    llm = RunnableLambda(lambda p: FakeLLM("Recurring coolant leaks; inspect the cooling system.").invoke(p))
    out = main.analyze("Deep analysis for vehicle 55", llm=llm, json_llm=json_llm)
    assert out["vehicle_id"] == 55 and len(out["parsed_vehicle_data"]) == 5
    assert "coolant" in out["analysis"]


def test_analyze_without_vehicle_id():
    from langchain_core.runnables import RunnableLambda

    json_llm = RunnableLambda(lambda p: FakeLLM('{"vehicle_id": null, "vehicle_name": "Hilux"}').invoke(p))
    with pytest.raises(ValueError):
        main.analyze("How is the Hilux doing?", llm=json_llm, json_llm=json_llm)
