import argparse
import asyncio
import sys
import json
import dataclasses
from typing import Any

import apple_fm_sdk as fm

def map_json_schema_type(schema: dict) -> type:
    t = schema.get("type", "string")
    if t == "string": return str
    if t == "integer": return int
    if t == "number": return float
    if t == "boolean": return bool
    if t == "array":
        items_schema = schema.get("items", {})
        item_type = map_json_schema_type(items_schema)
        return list[item_type]
    if t == "object":
        return create_dynamic_class("NestedObject", schema)
    return str

def create_dynamic_class(name: str, schema: dict) -> type:
    annotations = {}
    class_attrs = {}

    for prop_name, prop_schema in schema.get("properties", {}).items():
        py_type = map_json_schema_type(prop_schema)
        annotations[prop_name] = py_type
        
        guide_kwargs = {}
        if "description" in prop_schema:
            guide_kwargs["description"] = prop_schema["description"]
        
        # Add basic constraints if present
        if "minimum" in prop_schema and "maximum" in prop_schema:
            guide_kwargs["range"] = (prop_schema["minimum"], prop_schema["maximum"])

        if guide_kwargs:
            desc = guide_kwargs.pop("description", None)
            if desc:
                class_attrs[prop_name] = fm.guide(desc, **guide_kwargs)
            else:
                class_attrs[prop_name] = fm.guide(**guide_kwargs)
    
    class_attrs["__annotations__"] = annotations
    cls = type(name, (object,), class_attrs)
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
            DynamicClass = create_dynamic_class("GeneratedObject", schema)
            response = await session.respond(query, generating=DynamicClass)
            
            # Print as JSON if possible
            if dataclasses.is_dataclass(response):
                print(json.dumps(dataclasses.asdict(response), indent=2))
            elif hasattr(response, "__dict__"):
                # Clean up any inner objects recursively for dict
                def clean_dict(obj: Any) -> Any:
                    if hasattr(obj, "__dict__"):
                        return {k: clean_dict(v) for k, v in vars(obj).items()}
                    if isinstance(obj, list):
                        return [clean_dict(x) for x in obj]
                    return obj
                print(json.dumps(clean_dict(response), indent=2))
            else:
                print(response)
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
