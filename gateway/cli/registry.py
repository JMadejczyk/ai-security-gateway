"""Subcommand registry: each CLI module contributes one top-level subcommand.

A module exposes ``register(subparsers: Subcommands) -> None``. It adds its parser(s) and
sets ``run`` on each leaf parser (``parser.set_defaults(run=run)``), where ``run`` is
``async def run(args: argparse.Namespace, client: OperatorClient) -> int``. To add a
subcommand, import its module here and append it to `COMMANDS`.
"""

import argparse
from collections.abc import Awaitable, Callable
from types import ModuleType
from typing import Final

from gateway.cli import approvals, kill, pin
from gateway.cli.client import OperatorClient

type Subcommands = argparse._SubParsersAction[argparse.ArgumentParser]  # pyright: ignore[reportPrivateUsage] -- argparse's public add_subparsers() returns this type
type Run = Callable[[argparse.Namespace, OperatorClient], Awaitable[int]]

COMMANDS: Final[tuple[ModuleType, ...]] = (approvals, kill, pin)
