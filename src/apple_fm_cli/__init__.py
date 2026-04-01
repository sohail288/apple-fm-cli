import argparse
import asyncio
import dataclasses
import html
import json
import re
import sys
from typing import Any

import httpx

import apple_fm_sdk as fm


def map_json_schema_to_type_and_guide(
    prop_name: str, prop_schema: dict[str, Any]
) -> tuple[type, Any]:
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
        py_type = list[item_type]  # type: ignore
    elif t == "object":
        py_type = create_dynamic_dataclass(f"{prop_name.capitalize()}Type", prop_schema)

    guide_kwargs: dict[str, Any] = {}
    if "description" in prop_schema:
        guide_kwargs["description"] = prop_schema["description"]
    if "minimum" in prop_schema and "maximum" in prop_schema:
        guide_kwargs["range"] = (prop_schema["minimum"], prop_schema["maximum"])

    if guide_kwargs:
        desc = guide_kwargs.pop("description", "")
        return py_type, fm.guide(desc, **guide_kwargs)

    return py_type, dataclasses.MISSING


def create_dynamic_dataclass(name: str, schema: dict[str, Any]) -> type:
    fields: list[Any] = []
    for prop_name, prop_schema in schema.get("properties", {}).items():
        py_type, guide = map_json_schema_to_type_and_guide(prop_name, prop_schema)
        if guide is not dataclasses.MISSING:
            fields.append((prop_name, py_type, guide))
        else:
            fields.append((prop_name, py_type))

    cls = dataclasses.make_dataclass(name, fields)
    return fm.generable(cls)  # type: ignore


@fm.generable
@dataclasses.dataclass
class BashParams:
    command: str = fm.guide("The shell command to run")


class BashTool(fm.Tool):  # type: ignore
    name = "bash"
    description = (
        "Executes a shell command on the user's local machine and returns the output. "
        "Use this to read files, run scripts, or perform system operations."
    )

    @property
    def arguments_schema(self) -> fm.GenerationSchema:
        return BashParams.generation_schema()  # type: ignore

    async def call(self, args: Any) -> str:
        command = getattr(args, "command", None)
        if command is None and hasattr(args, "value"):
            command = args.value(str, for_property="command")

        if not command:
            return "Error: Could not extract command from tool arguments."

        process = await asyncio.create_subprocess_shell(
            command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        out = ""
        if stdout:
            out += f"STDOUT:\n{stdout.decode()}\n"
        if stderr:
            out += f"STDERR:\n{stderr.decode()}\n"
        return out.strip() or "Command executed successfully with no output."


@fm.generable
@dataclasses.dataclass
class GoogleSearchParams:
    query: str = fm.guide("The search query")


class GoogleSearchTool(fm.Tool):  # type: ignore
    name = "google_search"
    description = (
        "Searches the web for current events, facts, or information. "
        "Returns snippets and the content of the top linked pages."
    )

    @property
    def arguments_schema(self) -> fm.GenerationSchema:
        return GoogleSearchParams.generation_schema()  # type: ignore

    async def call(self, args: Any) -> str:
        query = getattr(args, "query", None)
        if query is None and hasattr(args, "value"):
            query = args.value(str, for_property="query")

        if not query:
            return "Error: Could not extract query from tool arguments."

        user_agent = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
        headers = {"User-Agent": user_agent}

        async with httpx.AsyncClient(
            headers=headers, follow_redirects=True, timeout=10.0
        ) as client:
            try:
                # Search on DuckDuckGo Lite
                search_url = "https://lite.duckduckgo.com/lite/"
                response = await client.post(search_url, data={"q": query})
                response.raise_for_status()
                html_content = response.text

                # Extract links: first find tags with class 'result-link'
                a_tags = re.findall(
                    r'<a[^>]+class=["\']result-link["\'][^>]*>.*?</a>',
                    html_content,
                    re.DOTALL | re.IGNORECASE,
                )
                link_matches = []
                for a in a_tags:
                    href_match = re.search(r'href=["\'](.*?)["\']', a, re.IGNORECASE)
                    if href_match:
                        title = re.sub(r"<[^>]+>", "", a).strip()
                        link_matches.append((href_match.group(1), title))

                snippets = re.findall(
                    r'<td class=["\']result-snippet["\'][^>]*>(.*?)</td>',
                    html_content,
                    re.DOTALL | re.IGNORECASE,
                )

                if not link_matches:
                    return "No results found."

                # Prepare the final report
                output = [f"Search results for: {query}\n"]

                # Extract top 3 links
                top_links = []
                for href, title in link_matches[:3]:
                    if href.startswith("//"):
                        href = "https:" + href
                    elif href.startswith("/"):
                        href = "https://duckduckgo.com" + href
                    top_links.append((href, html.unescape(title)))

                # Fetch contents of top 3 links concurrently
                async def fetch_page(url: str, title: str) -> str:
                    try:
                        resp = await client.get(url, timeout=5.0)
                        if resp.status_code == 200:
                            # Extract body text (crude extraction)
                            text = re.sub(
                                r"<(script|style|header|footer|nav)[^>]*>.*?</\1>",
                                "",
                                resp.text,
                                flags=re.DOTALL | re.IGNORECASE,
                            )
                            text = re.sub(r"<[^>]+>", " ", text)
                            text = html.unescape(text)
                            text = re.sub(r"\s+", " ", text).strip()
                            return f"Source: {title} ({url})\nContent: {text[:1500]}..."
                        return (
                            f"Source: {title} ({url})\n"
                            f"Status: Failed to fetch (HTTP {resp.status_code})"
                        )
                    except Exception as e:
                        return f"Source: {title} ({url})\nStatus: Error ({e!s})"

                page_contents = await asyncio.gather(*(fetch_page(u, t) for u, t in top_links))

                output.append("Summaries from Top Pages:")
                output.extend(page_contents)

                if snippets:
                    output.append("\nSearch Snippets:")
                    for s in snippets[:3]:
                        output.append("- " + html.unescape(re.sub(r"<[^>]+>", "", s)).strip())

                return "\n\n".join(output)

            except Exception as e:
                return f"Search error: {e}"


AVAILABLE_TOOLS = {"bash": BashTool, "google_search": GoogleSearchTool}


async def run_query(
    query: str, output_format: str, output_schema_str: str | None, tools_str: str | None
) -> None:
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
                available = ", ".join(AVAILABLE_TOOLS.keys())
                print(f"Warning: Unknown tool '{t_name}'. Available: {available}", file=sys.stderr)

    session_kwargs: dict[str, Any] = {}
    if tool_instances:
        session_kwargs["tools"] = tool_instances

    session = fm.LanguageModelSession(**session_kwargs)

    try:
        if output_format == "json" and output_schema_str:
            schema = json.loads(output_schema_str)
            dynamic_class = create_dynamic_dataclass("GeneratedObject", schema)
            response = await session.respond(query, generating=dynamic_class)

            # Print as JSON using dataclasses.asdict
            if dataclasses.is_dataclass(response):
                print(json.dumps(dataclasses.asdict(response), indent=2))  # type: ignore
            else:
                # Fallback for non-dataclass objects (though our dynamic ones should be)
                def clean_obj(obj: Any) -> Any:
                    if dataclasses.is_dataclass(obj):
                        return dataclasses.asdict(obj)  # type: ignore
                    if isinstance(obj, list):
                        return [clean_obj(x) for x in obj]
                    if hasattr(obj, "__dict__"):
                        return {k: clean_obj(v) for k, v in vars(obj).items()}
                    return obj

                print(json.dumps(clean_obj(response), indent=2))
        else:
            response = await session.respond(query)
            # if we have tool usage, the response object from sdk usually formats correctly
            if hasattr(response, "text"):
                print(response.text)
            else:
                print(response)

    except Exception as e:
        print(f"Error generating response: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query Apple Intelligence via CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # 'query' command (default behavior)
    query_parser = subparsers.add_parser("query", help="Query the model (default)")
    query_parser.add_argument("-q", "--query", type=str, required=True, help="The query to send")
    query_parser.add_argument(
        "--output",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format (text/json)",
    )
    query_parser.add_argument("--output-schema", type=str, help="JSON schema for output")
    query_parser.add_argument(
        "--tools", type=str, help="Comma-separated tools (bash,google_search)"
    )

    # 'server' command
    server_parser = subparsers.add_parser("server", help="Start an OpenAI-compatible server")
    server_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    server_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")

    # If no arguments or just -q ..., we should handle it gracefully for backwards compatibility
    # but the sub-parser makes it slightly different.
    # Let's check sys.argv and inject 'query' if first arg looks like -q or --query
    import sys

    if len(sys.argv) > 1 and sys.argv[1] in ("-q", "--query", "--output", "--tools"):
        sys.argv.insert(1, "query")

    args = parser.parse_args()

    if args.command == "query":
        if args.output == "json" and not args.output_schema:
            print("Error: --output-schema is required when --output is json", file=sys.stderr)
            sys.exit(1)
        asyncio.run(run_query(args.query, args.output, args.output_schema, args.tools))
    elif args.command == "server":
        from apple_fm_cli.server import run_server

        run_server(host=args.host, port=args.port)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
