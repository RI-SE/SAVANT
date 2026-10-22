"""dataprov helpers shared by SAVANT tools (needs dataprov and scenario-uncertainty)."""

import argparse
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from scenario_uncertainty import provenance

from savant_common.quality import origin_shares

SAVANT_NAMESPACE_PREFIX = "savant"
SAVANT_NAMESPACE_URI = "https://ri-se.github.io/SAVANT/ns#"
DECISION_LOG_FORMAT = "JSON"
ERROR_MODEL_FORMAT = provenance.output_format("model", 1)


def namespaces() -> dict:
    """Prefixes SAVANT steps use in custom properties."""
    return {SAVANT_NAMESPACE_PREFIX: SAVANT_NAMESPACE_URI, **provenance.namespaces()}


def arguments_string(parser: argparse.ArgumentParser, args: argparse.Namespace, secret: Iterable[str] = ()) -> str:
    """Every CLI option with a value (defaults included) as a reproducible argument string; secret dests omitted."""
    secret = set(secret)
    parts = []
    for action in parser._actions:
        if not action.option_strings or action.dest in secret or action.dest == "help":
            continue
        value = getattr(args, action.dest, None)
        flag = action.option_strings[0]
        if isinstance(action, argparse._StoreTrueAction):
            if value:
                parts.append(flag)
        elif isinstance(action, argparse._StoreFalseAction):
            if not value:
                parts.append(flag)
        elif value is not None:
            values = value if isinstance(value, (list, tuple)) else [value]
            parts.append(" ".join([flag, *map(str, values)]))
    return " ".join(parts)


def load_or_create_chain(path, **create_kwargs):
    """dataprov chain with the savant and su prefixes declared, also when the chain file already exists."""
    from dataprov import ProvenanceChain

    chain = ProvenanceChain.load_or_create(str(path), custom_namespaces=namespaces(), **create_kwargs)
    declare_namespaces(chain)
    return chain


def declare_namespaces(chain) -> None:
    """Declare the savant and su prefixes on a chain.

    dataprov applies custom_namespaces only when it creates a chain, and rejects undeclared prefixes.
    Same registry update as scenario_uncertainty.provenance.declare_namespace, for every prefix.
    """
    data = getattr(chain, "data", None)
    prefix_section = data.setdefault("prefix", chain.namespaces) if isinstance(data, dict) else None
    for prefix, uri in namespaces().items():
        chain.namespaces.setdefault(prefix, uri)
        if prefix_section is not None:
            prefix_section.setdefault(prefix, uri)


def origin_properties(counts: Counter, source: str = None) -> dict:
    """custom_properties of a step that wrote or used boxes: the su shares and the box count per origin."""
    properties = provenance.custom_properties(origin_shares(counts, source))
    properties[f"activity:{SAVANT_NAMESPACE_PREFIX}:origin_counts"] = dict(sorted(counts.items()))
    return properties


def existing(paths: Iterable) -> list:
    """The paths that exist, as strings (dataprov hashes every output)."""
    return [str(p) for p in paths if p and Path(p).exists()]


def edit_session(openlabel_path) -> dict:
    """What a provenance step for an edit session can say about a saved file.

    {"tool_version", "annotators", "custom_properties"}; tool_version is None for files saved by an edit
    that did not record its version.
    """
    from savant_common.openlabel import EDIT_METADATA_KEY, load_openlabel

    openlabel = load_openlabel(openlabel_path)
    metadata = openlabel.metadata.model_extra or {}
    annotators = openlabel.human_annotators()
    properties = origin_properties(openlabel.origin_counts(), source="savant_edit")
    properties[f"activity:{SAVANT_NAMESPACE_PREFIX}:annotators"] = annotators
    return {"tool_version": (metadata.get(EDIT_METADATA_KEY) or {}).get("version"),
            "annotators": annotators, "custom_properties": properties}
