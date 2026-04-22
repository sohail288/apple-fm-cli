import argparse
import asyncio
import dataclasses
import html
import json
import logging
import re
import sys
from collections.abc import Sequence
from typing import Any, cast

import httpx

import apple_fm_sdk as fm

logger = logging.getLogger(__name__)


def map_json_schema_to_type_and_guide(
    prop_name: str, prop_schema: dict[str, Any]
) -> tuple[type, Any]:
    t = prop_schema.get("type", "string")
    final_type: type = str

    if t == "integer":
        final_type = int
    elif t == "number":
        final_type = float
    elif t == "boolean":
        final_type = bool
    elif t == "array":
        items_schema = prop_schema.get("items", {})
        map_json_schema_to_type_and_guide(f"{prop_name}_item", items_schema)
        final_type = list
    elif t == "object":
        final_type = create_dynamic_dataclass(f"{prop_name.capitalize()}Type", prop_schema)

    guide_kwargs: dict[str, Any] = {}
    if "description" in prop_schema:
        guide_kwargs["description"] = prop_schema["description"]
    if "minimum" in prop_schema and "maximum" in prop_schema:
        cast(Any, guide_kwargs)["range"] = (prop_schema["minimum"], prop_schema["maximum"])

    if guide_kwargs:
        desc_val = str(guide_kwargs.pop("description", ""))
        guide_val = fm.guide(desc_val, **guide_kwargs)
        return final_type, guide_val

    return final_type, dataclasses.MISSING


def create_dynamic_dataclass(name: str, schema: dict[str, Any]) -> type[fm.Generable]:
    fields: list[Any] = []
    for prop_name, prop_schema in schema.get("properties", {}).items():
        target_type, target_guide = map_json_schema_to_type_and_guide(prop_name, prop_schema)
        if target_guide is not dataclasses.MISSING:
            fields.append((prop_name, target_type, target_guide))
        else:
            fields.append((prop_name, target_type))

    dynamic_cls = dataclasses.make_dataclass(name, fields)
    return cast(type, fm.generable(dynamic_cls))


@fm.generable
@dataclasses.dataclass
class BashParams:
    command: str = fm.guide("The shell command to run")  # type: ignore[assignment]


def extract_tool_argument(args: Any, *property_names: str) -> Any:
    for property_name in property_names:
        value = getattr(args, property_name, None)
        if value is not None:
            return value

    if hasattr(args, "value"):
        for property_name in property_names:
            try:
                return args.value(str, for_property=property_name)
            except Exception as error:
                logger.debug("Tool argument lookup failed for %s: %s", property_name, error)
                continue

    return None


class BashTool(fm.Tool):
    name = "bash"
    description = (
        "Executes a shell command on the user's local machine and returns the output. "
        "Use this to read files, run scripts, or perform system operations."
    )

    @property
    def arguments_schema(self) -> fm.GenerationSchema:
        return BashParams.generation_schema()  # type: ignore[attr-defined, no-any-return]

    async def call(self, args: Any) -> str:
        extracted_command = extract_tool_argument(args, "command")

        if not extracted_command:
            return "Error: Could not extract command from tool arguments."

        process = await asyncio.create_subprocess_shell(
            extracted_command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
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
    query: str = fm.guide("The search query")  # type: ignore[assignment]


class GoogleSearchTool(fm.Tool):
    name = "google_search"
    description = (
        "Searches the web for current events, facts, or information. "
        "Returns snippets and the content of the top linked pages."
    )

    @property
    def arguments_schema(self) -> fm.GenerationSchema:
        return GoogleSearchParams.generation_schema()  # type: ignore[attr-defined, no-any-return]

    async def call(self, args: Any) -> str:
        final_query = extract_tool_argument(args, "query", "query_text")

        if not final_query:
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
                search_url = "https://lite.duckduckgo.com/lite/"
                response = await client.post(search_url, data={"q": final_query})
                response.raise_for_status()
                html_content = response.text

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

                output = [f"Search results for: {final_query}\n"]

                top_links = []
                for href, title in link_matches[:3]:
                    if href.startswith("//"):
                        href = "https:" + href
                    elif href.startswith("/"):
                        href = "https://duckduckgo.com" + href
                    top_links.append((href, html.unescape(title)))

                async def fetch_page(url: str, title: str) -> str:
                    try:
                        resp = await client.get(url, timeout=5.0)
                        if resp.status_code == 200:
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
        session_kwargs["instructions"] = (
            "Use the available tools whenever the answer depends on filesystem, shell, or web "
            "information. After tool use, answer directly from the tool results."
        )

    session = fm.LanguageModelSession(**session_kwargs)

    try:
        if output_format == "json" and output_schema_str:
            schema_data = json.loads(output_schema_str)
            final_gen_type = create_dynamic_dataclass("GeneratedObject", schema_data)
            generated_content = await session.respond(
                query,
                schema=final_gen_type.generation_schema(),
            )
            print(json.dumps(generated_content.to_dict(), indent=2))
        else:
            final_resp = await session.respond(query)
            if hasattr(final_resp, "text"):
                print(final_resp.text)
            else:
                print(final_resp)

    except Exception as e:
        print(f"Error generating response: {e}", file=sys.stderr)
        sys.exit(1)


def build_legacy_query_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query Apple Intelligence via CLI")
    parser.add_argument("-q", "--query", required=True, help="The prompt to send to the model")
    parser.add_argument(
        "--output",
        dest="format",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )
    parser.add_argument(
        "--output-schema",
        dest="schema",
        help="JSON schema for guided generation (if output=json)",
    )
    parser.add_argument("--tools", help="Comma-separated list of tools to enable")
    return parser


def parse_cli_args(argv: Sequence[str], root_parser: argparse.ArgumentParser) -> argparse.Namespace:
    if argv and argv[0] in ("server", "query", "embeddings"):
        return root_parser.parse_args(list(argv))

    legacy_parser = build_legacy_query_parser()
    args = legacy_parser.parse_args(list(argv))
    args.command = "query"
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description="Query Apple Intelligence via CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    query_parser = subparsers.add_parser("query", help="Query the model")
    query_parser.add_argument("query", help="The prompt to send to the model")
    query_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )
    query_parser.add_argument("--schema", help="JSON schema for guided generation (if format=json)")
    query_parser.add_argument("--tools", help="Comma-separated list of tools to enable")

    server_parser = subparsers.add_parser("server", help="Start the OpenAI-compatible API server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")  # noqa: S104
    server_parser.add_argument("--port", type=int, default=8000, help="Port to listen on")

    embeddings_parser = subparsers.add_parser("embeddings", help="Generate sentence embeddings")
    embeddings_parser.add_argument("text", help="The text to embed")

    args = parse_cli_args(sys.argv[1:], parser)

    if args.command == "query":
        asyncio.run(run_query(args.query, args.format, args.schema, args.tools))
    elif args.command == "embeddings":
        try:
            vector = fm.get_sentence_embedding(args.text)
            print(json.dumps(vector))
        except Exception as e:
            print(f"Error generating embeddings: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.command == "server":
        from apple_fm_cli.server import run_server

        run_server(host=args.host, port=args.port)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
