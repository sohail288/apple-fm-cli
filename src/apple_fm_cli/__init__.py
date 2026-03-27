import argparse
import asyncio
import sys
import json
import dataclasses
import urllib.request
import urllib.parse
import re
import html
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

@fm.generable
@dataclasses.dataclass
class BashParams:
    command: str = fm.guide("The shell command to run")

class BashTool(fm.Tool):
    name = "bash"
    description = "Executes a shell command on the user's local machine and returns the output. Use this to read files, run scripts, or perform system operations."

    @property
    def arguments_schema(self) -> fm.GenerationSchema:
        return BashParams.generation_schema()

    async def call(self, args: Any) -> str:
        command = getattr(args, "command", None)
        if command is None and hasattr(args, "value"):
            command = args.value(str, for_property="command")
        
        if not command:
            return "Error: Could not extract command from tool arguments."
            
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        out = ""
        if stdout: out += f"STDOUT:\n{stdout.decode()}\n"
        if stderr: out += f"STDERR:\n{stderr.decode()}\n"
        return out.strip() or "Command executed successfully with no output."

@fm.generable
@dataclasses.dataclass
class GoogleSearchParams:
    query: str = fm.guide("The search query")

class GoogleSearchTool(fm.Tool):
    name = "google_search"
    description = "Searches the web for current events, facts, or information and returns text snippets."

    @property
    def arguments_schema(self) -> fm.GenerationSchema:
        return GoogleSearchParams.generation_schema()

    async def call(self, args: Any) -> str:
        query = getattr(args, "query", None)
        if query is None and hasattr(args, "value"):
            query = args.value(str, for_property="query")
            
        if not query:
            return "Error: Could not extract query from tool arguments."
            
        try:
            url = "https://lite.duckduckgo.com/lite/"
            data = urllib.parse.urlencode({'q': query}).encode('utf-8')
            req = urllib.request.Request(url, data=data, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, urllib.request.urlopen, req)
            html_content = response.read().decode('utf-8')
            
            # Use flexible regex for both snippets and links to handle quote variance
            snippets = re.findall(r'<td class=["\']result-snippet["\'][^>]*>(.*?)</td>', html_content, re.DOTALL | re.IGNORECASE)
            
            if not snippets:
                # Fallback to link titles if snippets aren't found
                links = re.findall(r'<a[^>]+class=["\']result-link["\'][^>]*>(.*?)</a>', html_content, re.DOTALL | re.IGNORECASE)
                if not links:
                    return "No results found."
                results = [html.unescape(re.sub(r'<[^>]+>', '', l)).strip() for l in links[:5]]
            else:
                results = [html.unescape(re.sub(r'<[^>]+>', '', s)).strip() for s in snippets[:5]]
                
            return "\n\n".join(results)
        except Exception as e:
            return f"Search error: {e}"

AVAILABLE_TOOLS = {
    "bash": BashTool,
    "google_search": GoogleSearchTool
}

async def run_query(query: str, output_format: str, output_schema_str: str | None, tools_str: str | None) -> None:
    model = fm.SystemLanguageModel()
    
    is_available, reason = model.is_available()
    if not is_available:
        print(f"Error: Foundation Models not available: {reason}", file=sys.stderr)
        sys.exit(1)

    tool_instances = []
    if tools_str:
        for t_name in tools_str.split(","):
            t_name = t_name.strip()
            if t_name in AVAILABLE_TOOLS:
                tool_instances.append(AVAILABLE_TOOLS[t_name]())
            else:
                print(f"Warning: Unknown tool '{t_name}'. Available: {', '.join(AVAILABLE_TOOLS.keys())}", file=sys.stderr)

    session_kwargs = {}
    if tool_instances:
        session_kwargs["tools"] = tool_instances

    session = fm.LanguageModelSession(**session_kwargs)
    
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
            # if we have tool usage, the response object from sdk usually formats correctly via __str__ or .text
            if hasattr(response, "text"):
                print(response.text)
            else:
                print(response)
            
    except Exception as e:
        print(f"Error generating response: {e}", file=sys.stderr)
        sys.exit(1)

def main() -> None:
    parser = argparse.ArgumentParser(description="Query Apple Intelligence via CLI")
    parser.add_argument("-q", "--query", type=str, required=True, help="The query to send to the model")
    parser.add_argument("--output", type=str, choices=["text", "json"], default="text", help="The output format (text or json)")
    parser.add_argument("--output-schema", type=str, help="JSON string defining the schema for json output")
    parser.add_argument("--tools", type=str, help="Comma-separated list of tools to enable (e.g., 'bash,google_search')")
    args = parser.parse_args()

    if args.output == "json" and not args.output_schema:
        print("Error: --output-schema is required when --output is json", file=sys.stderr)
        sys.exit(1)

    asyncio.run(run_query(args.query, args.output, args.output_schema, args.tools))

if __name__ == "__main__":
    main()
