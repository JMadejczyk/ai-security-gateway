"""Prices from the policy's ``pricing`` table, in nano-USD.

Arithmetic is decimal on the prices as written (``0.0005`` stays exactly ``0.0005``), and a
cost is rounded up to the next nano-USD, so a reservation never undershoots and the rounding
stays far below any configurable limit. A model without a pricing entry costs nothing; its
tokens and GPU time are still metered.
"""

from collections.abc import Mapping
from decimal import ROUND_CEILING, Decimal
from typing import Final

from gateway.budget.model import MS_PER_SECOND, NANO_USD_PER_USD
from gateway.policy.schema import ModelPrice

_FREE: Final = ModelPrice()
_TOKENS_PER_PRICE_UNIT: Final = 1_000


class CostModel:
    """Looks up one model's price and turns usage into nano-USD."""

    def __init__(self, pricing: Mapping[str, ModelPrice]) -> None:
        self._pricing = pricing

    def price(self, model: str | None) -> ModelPrice:
        if model is None:
            return _FREE
        return self._pricing.get(model, _FREE)

    def token_cost(self, model: str | None, *, prompt: int, completion: int) -> int:
        price = self.price(model)
        usd = (
            prompt * _exact(price.prompt_per_1k) + completion * _exact(price.completion_per_1k)
        ) / _TOKENS_PER_PRICE_UNIT
        return _nano(usd)

    def gpu_cost(self, model: str | None, *, gpu_ms: int) -> int:
        return _nano(gpu_ms * _exact(self.price(model).gpu_second) / MS_PER_SECOND)

    def affordable_gpu_ms(self, model: str | None, *, budget_nano_usd: int) -> int | None:
        """The most GPU milliseconds ``budget_nano_usd`` pays for; None when GPU time is free."""
        price = self.price(model).gpu_second
        if price <= 0:
            return None
        if budget_nano_usd <= 0:
            return 0
        per_ms = _exact(price) * NANO_USD_PER_USD / MS_PER_SECOND
        gpu_ms = int(Decimal(budget_nano_usd) / per_ms)  # floor: rounding up could overspend
        return gpu_ms if self.gpu_cost(model, gpu_ms=gpu_ms) <= budget_nano_usd else gpu_ms - 1


def _exact(price: float) -> Decimal:
    return Decimal(repr(price))  # the shortest decimal that round-trips: the value as written


def _nano(usd: Decimal) -> int:
    return int((usd * NANO_USD_PER_USD).to_integral_value(rounding=ROUND_CEILING))
