"""A multiplexed profile that adopted another profile's live MCP connection keeps that
server's tools across the owner's park -> revival cycle.

Chase gateway (one process, profiles ``default`` + ``chase-expert``): ``default`` connects
``purchase_orders_read`` / ``purchase_orders_handoff`` and ``chase-expert`` adopts them (same
route, same credentials). When the 18030 backend restarted, both connections parked, which
deregisters their tools from every profile overlay; on revival the owner re-registered only
into its own scope, so every later chase-expert session's catalog lacked both servers while
``documents_search`` / ``shottys_vip_reports`` (never parked) stayed visible.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_constants import hermes_home_key, reset_hermes_home_override, set_hermes_home_override


def _cfg(url: str, token: str, timeout: int = 90) -> dict:
    return {"url": url, "protocol": "legacy", "timeout": timeout, "connect_timeout": 15,
            "supports_parallel_tool_calls": False, "trust": "full",
            "headers": {"Authorization": f"Bearer {token}"},
            "tools": {"resources": False, "prompts": False}}


DEFAULT_SERVERS = {
    "documents_search": _cfg("http://127.0.0.1:18020/mcp/search/", "docs"),
    "documents_write": _cfg("http://127.0.0.1:18020/mcp/write/", "docs-write"),
    "purchase_orders_read": _cfg("http://127.0.0.1:18030/chase/read/mcp/", "po-read"),
    "shottys_vip_reports": _cfg("http://127.0.0.1:18040/mcp", "vip"),
    "purchase_orders_handoff": _cfg("http://127.0.0.1:18030/chase/actions/mcp/", "po-handoff", 960),
}
CHASE_SERVERS = {
    "documents_search": _cfg("http://127.0.0.1:18020/mcp/search/", "docs"),
    "purchase_orders_read": _cfg("http://127.0.0.1:18030/chase/read/mcp/", "po-read"),
    "shottys_vip_reports": _cfg("http://127.0.0.1:18040/mcp", "vip"),
    "purchase_orders_handoff": _cfg("http://127.0.0.1:18030/chase/actions/mcp/", "po-handoff", 420),
}
TOOLS = {"documents_search": ["read_document"], "documents_write": ["apply_finding"],
         "purchase_orders_read": ["list_records_for_table", "get_order_email_status"],
         "shottys_vip_reports": ["run_sql"], "purchase_orders_handoff": ["process_purchase_order"]}


def _mcp_tool(name):
    return SimpleNamespace(name=name, description="d", inputSchema={"type": "object", "properties": {}},
                           annotations=None)


@pytest.fixture
def chase_gateway(tmp_path, monkeypatch):
    """Multiplex on, clean MCP ledgers, homes ``default`` and ``chase-expert`` with their config."""
    import json
    import tools.mcp_tool as core
    from tools import mcp_tool_config as _config
    from tools import mcp_tool_registration as _registration

    homes = {"default": tmp_path / "default", "chase": tmp_path / "default" / "profiles" / "chase-expert"}
    for which, home in homes.items():
        home.mkdir(parents=True)
        servers = DEFAULT_SERVERS if which == "default" else CHASE_SERVERS
        (home / "config.yaml").write_text(json.dumps({"mcp_servers": servers}))  # JSON is YAML
    monkeypatch.setattr("agent.secret_scope.is_multiplex_active", lambda: True)
    monkeypatch.setattr(core, "_ensure_mcp_sdk", lambda: True)
    monkeypatch.setattr(_config, "_filter_suspicious_mcp_servers", lambda servers: servers)
    monkeypatch.setattr(_config, "_load_mcp_config",
                        lambda: dict(CHASE_SERVERS if "chase-expert" in hermes_home_key() else DEFAULT_SERVERS))
    monkeypatch.setattr(_registration, "_write_schema_cache", lambda *a, **k: None)
    ledgers = ("_servers", "_server_scope_keys", "_server_tool_scopes", "_server_connecting",
               "_server_connect_errors", "_server_connect_retry_after", "_server_connect_failures",
               "_lazy_server_configs", "_mcp_tool_server_names", "_orphaned_adopters",
               "_parallel_safe_servers", "_server_trust_levels", "_tool_read_only_hints")
    saved = {n: type(getattr(core, n))(getattr(core, n)) for n in ledgers}
    for n in ledgers:
        getattr(core, n).clear()
    tokens = []

    def enter(which):
        tokens.append(set_hermes_home_override(homes[which]))
        return hermes_home_key(homes[which])

    yield enter
    from tools.registry import registry
    for name, tool_names in TOOLS.items():
        for tool in tool_names:
            for home in homes.values():
                registry.deregister(f"mcp__{name}__{tool}", scope=hermes_home_key(home))
    for token in reversed(tokens):
        reset_hermes_home_override(token)
    for n in ledgers:
        getattr(core, n).clear()
        getattr(core, n).update(saved[n])


def _visible_servers() -> set:
    """MCP servers whose tools the current profile's sessions would see (registered + check_fn)."""
    from tools.registry import registry
    visible = set()
    for name in TOOLS:
        entries = [registry.get_entry(t) for t in registry.get_tool_names_for_toolset(f"mcp-{name}")]
        if entries and all(e is not None and (e.check_fn is None or e.check_fn()) for e in entries):
            visible.add(name)
    return visible


@pytest.mark.parametrize("revival", ["reconnect", "list_changed"])
def test_adopting_profile_keeps_purchase_order_tools_after_owner_park_and_revival(chase_gateway, revival):
    import asyncio
    from tools import mcp_tool_discovery as disc
    from tools.mcp_tool import MCPServerTask

    # Gateway boot, default pass: default owns all five connections.
    chase_gateway("default")
    owned = {}

    def default_pass(new_servers):
        for name, cfg in new_servers.items():
            server = MCPServerTask(name)
            server._config = dict(cfg)
            server._tools = [_mcp_tool(t) for t in TOOLS[name]]
            server.session = MagicMock()
            identity = getattr(disc._registration, "_adopter_identity_digest", None)
            if identity is not None:  # trees that publish the resolved connect identity
                server._resolved_identity = identity(name, cfg)
            owned[name] = server
            disc._adopt_server(name, server)
            server._registered_tool_names = disc._registration._register_server_tools(name, server, cfg)

    with patch.object(disc, "_run_discovery_pass", default_pass), \
            patch.object(disc._loop, "_ensure_mcp_loop", lambda: None):
        disc.register_mcp_servers(dict(DEFAULT_SERVERS))
    assert _visible_servers() == set(DEFAULT_SERVERS)

    # chase-expert pass: identical route + credentials, so it adopts all four, connects nothing.
    chase_gateway("chase")
    with patch.object(disc, "_run_discovery_pass", lambda new: pytest.fail(f"connected {sorted(new)}")), \
            patch.object(disc._loop, "_ensure_mcp_loop", lambda: None):
        disc.register_mcp_servers(dict(CHASE_SERVERS))
    assert _visible_servers() == set(CHASE_SERVERS)

    # The 18030 backend restarts: both purchase-order connections park, then revive. Both run on
    # the MCP loop in the owner's (default) context.
    chase_gateway("default")
    for name in ("purchase_orders_read", "purchase_orders_handoff"):
        server = owned[name]
        server._deregister_tools()  # what _park() does
        server.session = MagicMock(list_tools=AsyncMock(
            return_value=SimpleNamespace(tools=[_mcp_tool(t) for t in TOOLS[name]], nextCursor=None)))
        if revival == "reconnect":  # the reconnect's own tools/list re-publishes
            server._tools = [_mcp_tool(t) for t in TOOLS[name]]
            server._register_discovered_tools_if_needed()
        else:  # the revived session's tools/list_changed notification gets there first
            asyncio.run(server._refresh_tools())
    assert _visible_servers() == set(DEFAULT_SERVERS)

    # A new chase-expert session sees every server its config names, not just the never-parked ones.
    chase_gateway("chase")
    assert _visible_servers() == set(CHASE_SERVERS)
