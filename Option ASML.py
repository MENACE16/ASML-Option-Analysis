import math
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from datetime import datetime


# ---------------------------
# Black-Scholes functions
# ---------------------------
def N(x):
    """Cumulative normal distribution"""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def n(x):
    """Standard normal probability density function"""
    return math.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)


def black_scholes_price(S, K, r, q, sigma, T, option_type='call'):
    """Calculate European call/put option price using Black-Scholes"""
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive")

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == 'call':
        return S * math.exp(-q * T) * N(d1) - K * math.exp(-r * T) * N(d2)
    else:
        return K * math.exp(-r * T) * N(-d2) - S * math.exp(-q * T) * N(-d1)


def calculate_greeks(S, K, r, q, sigma, T, option_type='call'):
    """Calculate option Greeks"""
    if T <= 0 or sigma <= 0:
        return None

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    greeks = {}

    if option_type == 'call':
        greeks['delta'] = math.exp(-q * T) * N(d1)
        greeks['theta'] = (-S * n(d1) * sigma * math.exp(-q * T) / (2 * math.sqrt(T))
                           - r * K * math.exp(-r * T) * N(d2)
                           + q * S * math.exp(-q * T) * N(d1)) / 365
        greeks['rho'] = K * T * math.exp(-r * T) * N(d2) / 100
    else:
        greeks['delta'] = math.exp(-q * T) * (N(d1) - 1)
        greeks['theta'] = (-S * n(d1) * sigma * math.exp(-q * T) / (2 * math.sqrt(T))
                           + r * K * math.exp(-r * T) * N(-d2)
                           - q * S * math.exp(-q * T) * N(-d1)) / 365
        greeks['rho'] = -K * T * math.exp(-r * T) * N(-d2) / 100

    # Common Greeks
    greeks['gamma'] = n(d1) * math.exp(-q * T) / (S * sigma * math.sqrt(T))
    greeks['vega'] = S * math.exp(-q * T) * n(d1) * math.sqrt(T) / 100

    return greeks


# ---------------------------
# Get ASML stock data
# ---------------------------
ticker = 'ASML'
print(f"Fetching {ticker} data...")
data = yf.download(ticker, period='1y', interval='1d', auto_adjust=True, progress=False)

if data.empty:
    raise ValueError("No data returned. Check ticker symbol or network connection.")

S0 = float(data['Close'].iloc[-1])

# ---------------------------
# Compute historical volatility (annualized)
# ---------------------------
close_prices = data['Close']
returns = np.log(close_prices / close_prices.shift(1)).dropna()
sigma_annual = float(returns.std() * np.sqrt(252))

print(f"\n{'=' * 50}")
print(f"ASML OPTION ANALYSIS - {datetime.now().strftime('%Y-%m-%d')}")
print(f"{'=' * 50}")
print(f"Current Stock Price (S0): ${S0:.2f}")
print(f"Annualized Volatility:     {sigma_annual:.2%}")

# ---------------------------
# Option parameters
# ---------------------------
# Corrected: Put at-the-money or slightly OTM, Call slightly OTM
K_put = S0 * 0.95  # Put: 5% below current (protective put)
K_call = S0 * 1.05  # Call: 5% above current (upside bet)
r = 0.045  # Current risk-free rate (~4.5%)
q = 0.01  # ASML dividend yield (~1%)
T_days = 30
T = T_days / 365

# ---------------------------
# Calculate option prices and Greeks
# ---------------------------
put_price = black_scholes_price(S0, K_put, r, q, sigma_annual, T, 'put')
call_price = black_scholes_price(S0, K_call, r, q, sigma_annual, T, 'call')

put_greeks = calculate_greeks(S0, K_put, r, q, sigma_annual, T, 'put')
call_greeks = calculate_greeks(S0, K_call, r, q, sigma_annual, T, 'call')

print(f"\n{'PUT OPTION':-^50}")
print(f"Strike Price:  ${K_put:.2f}")
print(f"Option Price:  ${put_price:.2f}")
print(f"Delta:         {put_greeks['delta']:.4f}")
print(f"Gamma:         {put_greeks['gamma']:.4f}")
print(f"Theta:         ${put_greeks['theta']:.4f}/day")
print(f"Vega:          ${put_greeks['vega']:.4f}")

print(f"\n{'CALL OPTION':-^50}")
print(f"Strike Price:  ${K_call:.2f}")
print(f"Option Price:  ${call_price:.2f}")
print(f"Delta:         {call_greeks['delta']:.4f}")
print(f"Gamma:         {call_greeks['gamma']:.4f}")
print(f"Theta:         ${call_greeks['theta']:.4f}/day")
print(f"Vega:          ${call_greeks['vega']:.4f}")

# ---------------------------
# Implied Volatility Comparison
# ---------------------------
print(f"\n{'PARAMETERS':-^50}")
print(f"Risk-Free Rate:     {r:.2%}")
print(f"Dividend Yield:     {q:.2%}")
print(f"Days to Expiration: {T_days}")

# ---------------------------
# Visualization
# ---------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'ASML Option Analysis (S0=${S0:.2f}, σ={sigma_annual:.2%})', fontsize=14, fontweight='bold')

S_range = np.linspace(S0 * 0.7, S0 * 1.3, 200)

# 1. Option Prices
ax1 = axes[0, 0]
put_prices = [black_scholes_price(S, K_put, r, q, sigma_annual, T, 'put') for S in S_range]
call_prices = [black_scholes_price(S, K_call, r, q, sigma_annual, T, 'call') for S in S_range]
ax1.plot(S_range, put_prices, 'r-', label=f'Put (K=${K_put:.0f})', linewidth=2)
ax1.plot(S_range, call_prices, 'g-', label=f'Call (K=${K_call:.0f})', linewidth=2)
ax1.axvline(S0, color='blue', linestyle='--', alpha=0.5, label='Current Price')
ax1.set_xlabel('Stock Price ($)')
ax1.set_ylabel('Option Price ($)')
ax1.set_title('Option Price vs Stock Price')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Payoff Diagrams at Expiration
ax2 = axes[0, 1]
put_payoff = [max(K_put - S, 0) - put_price for S in S_range]
call_payoff = [max(S - K_call, 0) - call_price for S in S_range]
ax2.plot(S_range, put_payoff, 'r-', label='Put P&L', linewidth=2)
ax2.plot(S_range, call_payoff, 'g-', label='Call P&L', linewidth=2)
ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
ax2.axvline(S0, color='blue', linestyle='--', alpha=0.5)
ax2.set_xlabel('Stock Price at Expiration ($)')
ax2.set_ylabel('Profit/Loss ($)')
ax2.set_title('Payoff at Expiration')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Delta
ax3 = axes[1, 0]
put_deltas = [calculate_greeks(S, K_put, r, q, sigma_annual, T, 'put')['delta'] for S in S_range]
call_deltas = [calculate_greeks(S, K_call, r, q, sigma_annual, T, 'call')['delta'] for S in S_range]
ax3.plot(S_range, put_deltas, 'r-', label='Put Delta', linewidth=2)
ax3.plot(S_range, call_deltas, 'g-', label='Call Delta', linewidth=2)
ax3.axvline(S0, color='blue', linestyle='--', alpha=0.5)
ax3.set_xlabel('Stock Price ($)')
ax3.set_ylabel('Delta')
ax3.set_title('Delta vs Stock Price')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Volatility Sensitivity
ax4 = axes[1, 1]
sigma_range = np.linspace(0.1, 1.0, 100)
put_vols = [black_scholes_price(S0, K_put, r, q, sig, T, 'put') for sig in sigma_range]
call_vols = [black_scholes_price(S0, K_call, r, q, sig, T, 'call') for sig in sigma_range]
ax4.plot(sigma_range * 100, put_vols, 'r-', label='Put', linewidth=2)
ax4.plot(sigma_range * 100, call_vols, 'g-', label='Call', linewidth=2)
ax4.axvline(sigma_annual * 100, color='blue', linestyle='--', alpha=0.5, label='Current Vol')
ax4.set_xlabel('Volatility (%)')
ax4.set_ylabel('Option Price ($)')
ax4.set_title('Option Price vs Volatility')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n{'=' * 50}")
print("Analysis complete!")