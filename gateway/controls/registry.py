"""The controls the pipeline runs, in their fixed order.

Adding a guardrail is: implement `Control` in ``gateway/controls/<id>.py``, add its settings to
the policy schema if it has any, and register an instance here at composition time. The
registry checks each control against its catalog entry so a class cannot claim stages,
channels or a kind its catalog entry does not declare.

``authn`` and ``authz`` are catalog entries too, but the pipeline implements them itself
(token verification and the decision model), so they are never registered.
"""

from collections.abc import Iterable
from typing import Final

from gateway.core.catalog import control_spec
from gateway.core.interfaces import Control
from gateway.core.types import Channel, ControlKind, Stage

PIPELINE_STEPS: Final = frozenset({"authn", "authz"})


class ControlRegistry:
    """Deterministic controls run first (cheap), semantic ones after; registration order within."""

    def __init__(self, controls: Iterable[Control] = ()) -> None:
        self._controls: list[Control] = []
        for control in controls:
            self.register(control)

    def register(self, control: Control) -> None:
        if control.id in PIPELINE_STEPS:
            msg = f"{control.id!r} is a pipeline step, not a registrable control"
            raise ValueError(msg)
        spec = control_spec(control.id)
        if any(registered.id == control.id for registered in self._controls):
            msg = f"control {control.id!r} is already registered"
            raise ValueError(msg)
        if not control.stages <= spec.stages:
            msg = f"control {control.id!r} runs at stages its catalog entry does not declare"
            raise ValueError(msg)
        if control.kind is not spec.kind or control.mandatory != spec.mandatory:
            msg = f"control {control.id!r} disagrees with its catalog entry on kind or mandatory"
            raise ValueError(msg)
        self._controls.append(control)
        self._controls.sort(key=lambda c: c.kind is not ControlKind.DETERMINISTIC)  # stable

    def for_stage(self, stage: Stage, channel: Channel) -> tuple[Control, ...]:
        return tuple(
            control
            for control in self._controls
            if stage in control.stages and channel in control_spec(control.id).channels
        )

    def __len__(self) -> int:
        return len(self._controls)
