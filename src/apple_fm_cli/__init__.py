import argparse
import asyncio
import sys
import json
import dataclasses
from typing import Any

import apple_fm_sdk as fm

def map_json_schema_to_type_and_guide(prop_name: str, prop_schema: dict) -> tuple[type, Any]:
    t = prop_schema.get("type", "string")
    py_type: type = str
    
    if t == "integer":
        py_type = int
    elif t == "number":
        py_type = float
    elif t == "boolean":
        py_type = bool
    elif t == "array":
        items_schema = prop_schema.get("items", {})
        item_type, _ = map_json_schema_to_type_and_guide(f"{prop_name}_item", items_schema)
        py_type = list[item_type] # type: ignore
    elif t == "object":
        py_type = create_dynamic_dataclass(f"{prop_name.capitalize()}Type", prop_schema)
    
    guide_kwargs = {}
    if "description" in prop_schema:
        guide_kwargs["description"] = prop_schema["description"]
    if "minimum" in prop_schema and "maximum" in prop_schema:
        guide_kwargs["range"] = (prop_schema["minimum"], prop_schema["maximum"])

    if guide_kwargs:
        desc = guide_kwargs.pop("description", "")
        return py_type, fm.guide(desc, **guide_kwargs)
    
    return py_type, dataclasses.MISSING

def create_dynamic_dataclass(name: str, schema: dict) -> type:
    fields = []
    for prop_name, prop_schema in schema.get("properties", {}).items():
        py_type, guide = map_json_schema_to_type_and_guide(prop_name, prop_schema)
        if guide is not dataclasses.MISSING:
            fields.append((prop_name, py_type, guide))
        else:
            fields.append((prop_name, py_type))
            
    cls = dataclasses.make_dataclass(name, fields)
    return fm.generable(cls)

async def run_query(query: str, output_format: str, output_schema_str: str | None) -> None:
    model = fm.SystemLanguageModel()
    
    is_available, reason = model.is_available()
    if not is_available:
        print(f"Error: Foundation Models not available: {reason}", file=sys.stderr)
        sys.exit(1)

    session = fm.LanguageModelSession()
    
    try:
        if output_format == "json" and output_schema_str:
            schema = json.loads(output_schema_str)
            DynamicClass = create_dynamic_dataclass("GeneratedObject", schema)
            response = await session.respond(query, generating=DynamicClass)
            
            # Print as JSON using dataclasses.asdict
            if dataclasses.is_dataclass(response):
                print(json.dumps(dataclasses.asdict(response), indent=2))
            else:
                # Fallback for non-dataclass objects (though our dynamic ones should be)
                def clean_obj(obj: Any) -> Any:
                    if dataclasses.is_dataclass(obj):
                        return dataclasses.asdict(obj)
                    if isinstance(obj, list):
                        return [clean_obj(x) for x in obj]
                    if hasattr(obj, "__dict__"):
                        return {k: clean_obj(v) for k, v in vars(obj).items()}
                    return obj
                print(json.dumps(clean_obj(response), indent=2))
        else:
            response = await session.respond(query)
            print(response)
            
    except Exception as e:
        print(f"Error generating response: {e}", file=sys.stderr)
        sys.exit(1)

def main() -> None:
    parser = argparse.ArgumentParser(description="Query Apple Intelligence via CLI")
    parser.add_argument("-q", "--query", type=str, required=True, help="The query to send to the model")
    parser.add_argument("--output", type=str, choices=["text", "json"], default="text", help="The output format (text or json)")
    parser.add_argument("--output-schema", type=str, help="JSON string defining the schema for json output")
    args = parser.parse_args()

    if args.output == "json" and not args.output_schema:
        print("Error: --output-schema is required when --output is json", file=sys.stderr)
        sys.exit(1)

    asyncio.run(run_query(args.query, args.output, args.output_schema))

if __name__ == "__main__":
    main()
