"""Expose the EHR tool registry as an MCP server.

The same tool definitions the in-house agent uses become available to any MCP client
(Claude Code, Claude Desktop, other teams' agents). Only READ tools are exported:
MCP clients do not go through our ``ToolGuard`` approval flow, so WRITE tools stay
internal until there is an approval mechanism on this boundary as well.

    FHIR_BASE_URL=http://localhost:8080/fhir care-agent-mcp          # real HAPI server
    care-agent-mcp --offline                                          # bundled demo data
"""

from __future__ import annotations

import argparse

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from care_agent.config import Settings
from care_agent.fhir.client import FHIRClient
from care_agent.fhir.memory_server import InMemoryFHIR
from care_agent.tools.fhir_tools import build_fhir_tools
from care_agent.tools.registry import Permission, ToolRegistry

INSTRUCTIONS = (
    "Read-only access to a synthetic FHIR R4 EHR. Tools return de-identified summaries "
    "(no names, addresses or contact details). Always pass the patient_id you were given."
)


def build_mcp_server(registry: ToolRegistry) -> MCPServer:
    server = MCPServer("mini-care-agent-ehr", instructions=INSTRUCTIONS)
    for tool in registry:
        if tool.permission is not Permission.READ:
            continue
        server.add_tool(
            tool.fn,
            name=tool.name,
            description=tool.description,
            annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
        )
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--offline", action="store_true", help="serve bundled synthetic records")
    args = parser.parse_args()

    settings = Settings.from_env()
    transport = InMemoryFHIR() if args.offline else None
    client = FHIRClient(settings.fhir_base_url, transport=transport)
    build_mcp_server(build_fhir_tools(client)).run("stdio")


if __name__ == "__main__":
    main()
