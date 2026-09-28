"""NeuroQuant Options Lab.

Offline-first option research primitives for observed-chain backtests, realistic
execution modelling, volatility research, and paper-trade reconciliation.
"""

from .backtest import (
    VerticalBacktestConfig,
    backtest_bull_call_entries,
    summarize_vertical_backtest,
)
from .data import chain_at_or_before, load_option_chain_csv, normalize_option_chain
from .execution import (
    ExecutionConfig,
    combo_cash_flow,
    execution_price,
    is_liquid,
    leg_cash_flow,
)
from .models import BullCallSpread, OptionContract, OptionLeg, OptionQuote, OptionType, Side
from .paper import ShadowTrade, append_shadow_trade, reconciliation_report
from .pricing import Greeks, black_scholes_greeks, black_scholes_price, implied_volatility
from .providers import OptionDataProvider, PaperBroker, ProviderCapabilities, RECOMMENDED_CAPABILITY_PROFILES
from .research import bootstrap_trade_returns, chronological_trade_split, execution_stress_summary
from .strategies import CheapVolBullishConfig, build_cheap_vol_bullish_entries
from .selector import BullCallSelectionConfig, select_bull_call_spread
from .volatility import (
    attach_realized_volatility,
    iv_to_rv_ratio,
    long_volatility_candidate_mask,
    realized_volatility,
)

__all__ = [
    "OptionType",
    "Side",
    "OptionContract",
    "OptionQuote",
    "OptionLeg",
    "BullCallSpread",
    "Greeks",
    "black_scholes_price",
    "black_scholes_greeks",
    "implied_volatility",
    "ExecutionConfig",
    "execution_price",
    "leg_cash_flow",
    "combo_cash_flow",
    "is_liquid",
    "normalize_option_chain",
    "load_option_chain_csv",
    "chain_at_or_before",
    "realized_volatility",
    "attach_realized_volatility",
    "iv_to_rv_ratio",
    "long_volatility_candidate_mask",
    "BullCallSelectionConfig",
    "select_bull_call_spread",
    "VerticalBacktestConfig",
    "backtest_bull_call_entries",
    "summarize_vertical_backtest",
    "ShadowTrade",
    "append_shadow_trade",
    "reconciliation_report",
    "OptionDataProvider",
    "PaperBroker",
    "ProviderCapabilities",
    "RECOMMENDED_CAPABILITY_PROFILES",
    "CheapVolBullishConfig",
    "build_cheap_vol_bullish_entries",
    "chronological_trade_split",
    "bootstrap_trade_returns",
    "execution_stress_summary",
]
