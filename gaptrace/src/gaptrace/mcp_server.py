"""MCP server: the same tools, for any AI client. Written for mcp >= 2
(MCPServer); on mcp 1.x the class is mcp.server.fastmcp.FastMCP."""
from __future__ import annotations

from .workspace import open_toolbox


def build_server(data: str = "data"):
    from mcp.server.mcpserver import MCPServer

    tb = open_toolbox(data)
    server = MCPServer("gaptrace", instructions=(
        "Research memory built from papers. Tools return short digests and ids; "
        "ask for a note or passage by id only when you need more."))

    @server.tool()
    def search_notes(query: str, note_type: str = "", k: int = 0) -> str:
        """Find notes by keyword and meaning. Returns headlines with ids."""
        return tb.search_notes(query, note_type, k)

    @server.tool()
    def get_note(note_id: str, level: int = 1) -> str:
        """One note: headline (level 0) or summary with its links (level 1)."""
        return tb.get_note(note_id, level)

    @server.tool()
    def neighbours(note_id: str, link_type: str = "") -> str:
        """Notes linked to this one by verified links."""
        return tb.neighbours(note_id, link_type)

    @server.tool()
    def get_passage(passage_id: str) -> str:
        """One source passage, for exact wording or numbers."""
        return tb.get_passage(passage_id)

    @server.tool()
    def search_passages(query: str, k: int = 5) -> str:
        """Search prepared contextual passage metadata; fetch raw passages for evidence."""
        return tb.search_passages(query, k)

    @server.tool()
    def lookup_fact(method: str = "", dataset: str = "", metric: str = "") -> str:
        """Exact reported values from the facts table."""
        return tb.lookup_fact(method, dataset, metric)

    @server.tool()
    def find_gaps(k: int = 0, gap_type: str = "") -> str:
        """Ranked research gaps found in the map."""
        return tb.find_gaps(k, gap_type)

    @server.tool()
    def get_trail(gap_id: str) -> str:
        """The evidence trail behind one gap."""
        return tb.get_trail(gap_id)

    @server.tool()
    def search_literature(query: str, k: int = 5) -> str:
        """Search current scholarly abstracts; results and failures are recorded."""
        return tb.search_literature(query, k)

    @server.tool()
    def review_gaps(k: int = 3, gap_id: str = "", provider: str = "", as_of: str = "") -> str:
        """Check evidence, search for closure and assess feasibility, then rerank."""
        return tb.review_gaps(k, gap_id, provider, as_of)

    return server
