"""docs/architecture.md stays in step with the compose files and its rendered images.

The topology diagram (the first Mermaid block) must name every compose service and draw every
compose network as a subgraph whose title says `internal` exactly when compose marks it so.
The demo overlay (`demo/compose*.yml`) counts too. `make diagrams` writes the images.
"""

import re
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "architecture.md"
IMAGES = ROOT / "docs" / "img" / "architecture"
DIAGRAMS = ("topology", "pipeline", "thesis")  # the order of the Mermaid blocks in DOC

_MERMAID = re.compile(r"^```mermaid\n(.*?)^```$", re.MULTILINE | re.DOTALL)
_SUBGRAPH = re.compile(r'^\s*subgraph\s+([\w-]+)\["([^"]*)"\]', re.MULTILINE)


def _compose_files() -> list[Path]:
    return [ROOT / "docker-compose.yml", *sorted((ROOT / "demo").glob("compose*.yml"))]


def _compose() -> tuple[set[str], dict[str, bool]]:
    """Every service name, and every network name with its `internal` flag, across the files."""
    services: set[str] = set()
    networks: dict[str, bool] = {}
    for path in _compose_files():
        data = cast(dict[str, Any], yaml.safe_load(path.read_text(encoding="utf-8")))
        services.update(cast(dict[str, Any], data.get("services") or {}))
        for name, spec in cast(dict[str, Any], data.get("networks") or {}).items():
            networks[name] = bool(cast(dict[str, Any], spec or {}).get("internal", False))
    return services, networks


def _blocks() -> list[str]:
    return _MERMAID.findall(DOC.read_text(encoding="utf-8"))


def _mentions(text: str, name: str) -> bool:
    return re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", text) is not None


def test_doc_has_one_block_per_rendered_diagram() -> None:
    assert len(_blocks()) == len(DIAGRAMS)


def test_compose_files_parse_into_services_and_networks() -> None:
    services, networks = _compose()
    assert {"gateway", "agent", "mcp-fetch"} <= services
    assert networks["edge"] is True
    assert networks["ops"] is False


@pytest.mark.parametrize("service", sorted(_compose()[0]))
def test_topology_names_every_compose_service(service: str) -> None:
    assert _mentions(_blocks()[0], service), f"{service} is missing from the topology diagram"


@pytest.mark.parametrize(("network", "internal"), sorted(_compose()[1].items()))
def test_topology_draws_every_compose_network_with_its_internal_flag(
    network: str, internal: bool
) -> None:
    titles = dict(_SUBGRAPH.findall(_blocks()[0]))
    assert network in titles, f"{network} has no subgraph in the topology diagram"
    flag = re.search(r"\binternal\b", titles[network]) is not None
    assert flag is internal, f"{network}: compose internal={internal}, title {titles[network]!r}"


@pytest.mark.parametrize("name", DIAGRAMS)
@pytest.mark.parametrize("variant", ["", "-dark"])
@pytest.mark.parametrize("ext", ["svg", "png"])
def test_rendered_images_exist(name: str, variant: str, ext: str) -> None:
    image = IMAGES / f"{name}{variant}.{ext}"
    assert image.is_file(), f"{image.relative_to(ROOT)} missing: run `make diagrams`"
    assert image.stat().st_size > 1000
