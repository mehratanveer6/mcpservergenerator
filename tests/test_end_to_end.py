"""Generate a server from the Pet Clinic spec, start it over stdio and drive it with an MCP client."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from mcp import Client, StdioServerParameters

from mcpforge.generator import GenerateOptions, generate

sys.path.insert(0, str(Path(__file__).parent))
from mock_api import API_KEY, MockServer  # noqa: E402

SPEC = Path(__file__).resolve().parents[1] / "examples" / "petclinic.yaml"


def _params(server_dir: Path, base_url: str, policy: str) -> StdioServerParameters:
    env = {**os.environ, "API_BASE_URL": base_url, "API_KEY": API_KEY, "MCPFORGE_POLICY": policy}
    return StdioServerParameters(command=sys.executable, args=[str(server_dir / "server.py")], env=env)


def _text(result) -> str:
    return "".join(c.text for c in result.content if hasattr(c, "text"))


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    out = tmp_path_factory.mktemp("petclinic_server")
    generate(SPEC, GenerateOptions(output=out))
    return out


async def test_tools_annotations_and_calls(generated):
    with MockServer() as api:
        async with Client(_params(generated, api.base_url, "read_write")) as client:
            tools = {t.name: t for t in (await client.list_tools()).tools}
            # destructive operations are not registered under the default policy
            assert "delete_pet" not in tools and "cancel_appointment" not in tools
            assert {"list_pets", "get_pet", "create_pet", "search_pets", "search_operations", "call_operation"} <= set(tools)
            assert tools["list_pets"].annotations.read_only_hint is True
            assert tools["create_pet"].annotations.read_only_hint is False
            assert tools["search_pets"].annotations.read_only_hint is True  # POST classified as read

            pets = json.loads(_text(await client.call_tool("list_pets", {"species": "dog"})))
            assert [p["name"] for p in pets] == ["Bruno"]

            created = json.loads(_text(await client.call_tool("create_pet", {"name": "Kiwi", "species": "bird"})))
            assert created["id"] == 3

            found = json.loads(_text(await client.call_tool("search_pets", {"query": "kiwi"})))
            assert found[0]["species"] == "bird"

            refused = await client.call_tool("call_operation", {"operation_id": "deletePet", "arguments": {"petId": 1}})
            assert refused.is_error and "disabled under policy" in _text(refused)
            assert 1 in api.app.state.pets
        assert set(api.app.state.seen_keys) == {API_KEY}


async def test_full_policy_allows_destructive(generated):
    with MockServer() as api:
        async with Client(_params(generated, api.base_url, "full")) as client:
            tools = {t.name: t for t in (await client.list_tools()).tools}
            assert tools["delete_pet"].annotations.destructive_hint is True
            result = await client.call_tool("delete_pet", {"pet_id": 2})
            assert not result.is_error
            assert 2 not in api.app.state.pets


async def test_search_operations_routes_intent(generated):
    with MockServer() as api:
        async with Client(_params(generated, api.base_url, "read_write")) as client:
            hits = json.loads(_text(await client.call_tool("search_operations", {"query": "book a vet visit appointment"})))
            assert hits[0]["operation_id"] == "bookAppointment"
