#!/usr/bin/env bash
set -e

# ANSI Color codes for prettier output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

echo -e "${BOLD}--- Starting E2E Testing for apple-fm-cli ---${NC}"

# 1. Installation
echo -e "\n${BOLD}[1/4] Installing apple-fm-cli...${NC}"
./install.sh

# Path setup (ensure ~/.local/bin is used even if not in PATH yet)
CLI_BIN="$HOME/.local/bin/apple-fm-cli"
if ! command -v apple-fm-cli &> /dev/null; then
    echo "Warning: apple-fm-cli not found in PATH, using full path: $CLI_BIN"
else
    CLI_BIN="apple-fm-cli"
fi

# 2. Basic Text Query
echo -e "\n${BOLD}[2/4] Testing basic text query...${NC}"
OUTPUT=$( "$CLI_BIN" -q "What is 2 + 2?" )
echo "Output: $OUTPUT"
if [[ "$OUTPUT" == *"4"* ]]; then
    echo -e "${GREEN}PASS: Basic query returned correct result.${NC}"
else
    echo -e "${RED}FAIL: Basic query did not return expected result.${NC}"
    exit 1
fi

# 3. Structured JSON Output
echo -e "\n${BOLD}[3/4] Testing structured JSON output...${NC}"
SCHEMA='{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}'
JSON_OUTPUT=$( "$CLI_BIN" -q "Generate a cat" --output json --output-schema "$SCHEMA" )
echo "Output: $JSON_OUTPUT"

# Verify JSON with jq
if echo "$JSON_OUTPUT" | jq -e '.name and .age' > /dev/null; then
    echo -e "${GREEN}PASS: JSON output matches the schema and is valid.${NC}"
else
    echo -e "${RED}FAIL: JSON output is invalid or missing expected fields.${NC}"
    exit 1
fi

# 4. Tool Usage (Bash)
echo -e "\n${BOLD}[4/4] Testing tool usage (bash)...${NC}"
BASH_OUTPUT=$( "$CLI_BIN" -q "What files are in the current directory? List them exactly." --tools bash )
echo "Output: $BASH_OUTPUT"
if [[ "$BASH_OUTPUT" == *"pyproject.toml"* ]]; then
    echo -e "${GREEN}PASS: Bash tool correctly identified local files.${NC}"
else
    echo -e "${RED}FAIL: Bash tool did not find expected files.${NC}"
    exit 1
fi

echo -e "\n${BOLD}${GREEN}--- ALL E2E TESTS PASSED ---${NC}"
