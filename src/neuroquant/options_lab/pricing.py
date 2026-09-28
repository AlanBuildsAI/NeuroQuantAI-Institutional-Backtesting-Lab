"""Dependency-free Black-Scholes-Merton pricing and Greeks.

Historical backtests should prefer observed NBBO prices over theoretical prices
whenever market quotes are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, pi, sqrt

from .models import OptionType


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    theta_per_day: float
    vega_per_vol_point: float
    rho_per_rate_point: float


def _d1_d2(
    spot: float,
    strike: float,
    t_years: float,
    rate: float,
    volatility: float,
    dividend_yield: float,
) -> tuple[float, float]:
    if min(spot, strike, t_years, volatility) <= 0:
        raise ValueError("spot, strike, t_years and volatility must be positive")
    root_t = sqrt(t_years)
    d1 = (
        log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility * volatility) * t_years
    ) / (volatility * root_t)
    return d1, d1 - volatility * root_t


def black_scholes_price(
    spot: float,
    strike: float,
    t_years: float,
    rate: float,
    volatility: float,
    option_type: OptionType | str = OptionType.CALL,
    dividend_yield: float = 0.0,
) -> float:
    """Return BSM price per share."""

    kind = OptionType(option_type)
    if t_years <= 0:
        if kind is OptionType.CALL:
            return max(spot - strike, 0.0)
        return max(strike - spot, 0.0)
    d1, d2 = _d1_d2(
        spot, strike, t_years, rate, volatility, dividend_yield
    )
    discounted_spot = spot * exp(-dividend_yield * t_years)
    discounted_strike = strike * exp(-rate * t_years)
    if kind is OptionType.CALL:
        return discounted_spot * _norm_cdf(d1) - discounted_strike * _norm_cdf(d2)
    return discounted_strike * _norm_cdf(-d2) - discounted_spot * _norm_cdf(-d1)


def black_scholes_greeks(
    spot: float,
    strike: float,
    t_years: float,
    rate: float,
    volatility: float,
    option_type: OptionType | str = OptionType.CALL,
    dividend_yield: float = 0.0,
) -> Greeks:
    """Return common Greeks.

    Theta is dollars per share per calendar day.
    Vega and rho are dollars per share for a one percentage-point change.
    """

    kind = OptionType(option_type)
    d1, d2 = _d1_d2(
        spot, strike, t_years, rate, volatility, dividend_yield
    )
    root_t = sqrt(t_years)
    discount_q = exp(-dividend_yield * t_years)
    discount_r = exp(-rate * t_years)
    pdf = _norm_pdf(d1)

    if kind is OptionType.CALL:
        delta = discount_q * _norm_cdf(d1)
        theta_year = (
            -(spot * discount_q * pdf * volatility) / (2.0 * root_t)
            - rate * strike * discount_r * _norm_cdf(d2)
            + dividend_yield * spot * discount_q * _norm_cdf(d1)
        )
        rho = strike * t_years * discount_r * _norm_cdf(d2)
    else:
        delta = discount_q * (_norm_cdf(d1) - 1.0)
        theta_year = (
            -(spot * discount_q * pdf * volatility) / (2.0 * root_t)
            + rate * strike * discount_r * _norm_cdf(-d2)
            - dividend_yield * spot * discount_q * _norm_cdf(-d1)
        )
        rho = -strike * t_years * discount_r * _norm_cdf(-d2)

    gamma = discount_q * pdf / (spot * volatility * root_t)
    vega = spot * discount_q * pdf * root_t

    return Greeks(
        delta=float(delta),
        gamma=float(gamma),
        theta_per_day=float(theta_year / 365.0),
        vega_per_vol_point=float(vega / 100.0),
        rho_per_rate_point=float(rho / 100.0),
    )


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    t_years: float,
    rate: float,
    option_type: OptionType | str = OptionType.CALL,
    dividend_yield: float = 0.0,
    lower: float = 1e-4,
    upper: float = 5.0,
    tolerance: float = 1e-7,
    max_iterations: int = 200,
) -> float:
    """Solve implied volatility with bisection."""

    if market_price < 0:
        raise ValueError("market_price cannot be negative")
    if t_years <= 0:
        raise ValueError("implied volatility is undefined at or after expiration")

    low_price = black_scholes_price(
        spot, strike, t_years, rate, lower, option_type, dividend_yield
    )
    high_price = black_scholes_price(
        spot, strike, t_years, rate, upper, option_type, dividend_yield
    )
    if not (low_price - tolerance <= market_price <= high_price + tolerance):
        raise ValueError("market_price is outside the model solvable range")

    lo, hi = lower, upper
    for _ in range(max_iterations):
        mid = (lo + hi) / 2.0
        value = black_scholes_price(
            spot, strike, t_years, rate, mid, option_type, dividend_yield
        )
        if abs(value - market_price) <= tolerance:
            return float(mid)
        if value < market_price:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2.0)
