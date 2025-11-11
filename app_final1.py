# app_final.py — Deep Hedging Dashboard (Buehler et al., 2018 inspired)
# Streamlit UI with: Results • Trader Summary • Replay • Stress Test • γ-Frontier

import os, math, time
os.environ["KMP_WARNINGS"] = "0"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import streamlit as st

# -------------------- Page / device --------------------
st.set_page_config(page_title="Deep Hedging Dashboard", layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# -------------------- Utils --------------------
def set_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

def cvar_np(pnl, alpha=0.05):
    arr = np.sort(np.array(pnl))
    k = max(1, int(alpha * len(arr)))
    return -float(np.mean(arr[:k]))

# -------------------- Models --------------------
class MLPHedger(nn.Module):  # stock-only
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, S_t, tau_norm, prev_delta):
        x = torch.stack([S_t, tau_norm, prev_delta], dim=-1)
        return self.net(x).squeeze(-1)

class GRUHedger(nn.Module):  # stock + call
    def __init__(self, hidden=128, num_instr=2):
        super().__init__()
        self.gru = nn.GRU(input_size=3, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, num_instr)
    def forward(self, seq):
        h, _ = self.gru(seq)
        return self.head(h)   # [N,T,2]

# -------------------- Simulators --------------------
@torch.no_grad()
def sim_gbm(n_paths, steps, S0, mu, sigma, dt):
    S = torch.zeros(n_paths, steps+1, device=device, dtype=dtype)
    S[:,0] = S0
    drift = (mu - 0.5*sigma**2)*dt
    vol   = sigma*math.sqrt(dt)
    for t in range(steps):
        z = torch.randn(n_paths, device=device, dtype=dtype)
        S[:,t+1] = S[:,t]*torch.exp(drift + vol*z)
    return S

@torch.no_grad()
def sim_heston(n_paths, steps, S0, r, dt, kappa, theta, xi, rho, v0):
    S = torch.zeros(n_paths, steps+1, device=device, dtype=dtype); S[:,0]=S0
    v = torch.zeros_like(S); v[:,0]=v0
    sqrt1mr2 = math.sqrt(1-rho**2)
    dt_t = torch.tensor(dt, device=device, dtype=dtype)
    for t in range(steps):
        z1 = torch.randn(n_paths, device=device, dtype=dtype)
        z2 = rho*z1 + sqrt1mr2*torch.randn(n_paths, device=device, dtype=dtype)
        v_t = torch.clamp(v[:,t], min=1e-8)
        v[:,t+1] = torch.clamp(v_t + kappa*(theta-v_t)*dt_t + xi*torch.sqrt(v_t*dt_t)*z2, min=1e-8)
        S[:,t+1] = S[:,t]*torch.exp((r-0.5*v_t)*dt_t + torch.sqrt(v_t*dt_t)*z1)
    return S, v

# -------------------- Black–Scholes bits --------------------
def bs_call_delta(S, K, tau, r, sigma):
    tau   = torch.clamp(torch.as_tensor(tau,   device=S.device, dtype=S.dtype), min=1e-8)
    sigma = torch.clamp(torch.as_tensor(sigma, device=S.device, dtype=S.dtype), min=1e-8)
    d1 = (torch.log(S / K) + (r + 0.5*sigma**2)*tau) / (sigma * torch.sqrt(tau))
    return 0.5*(1+torch.erf(d1/math.sqrt(2.0)))

def bs_call_price(S, K, tau, r, sigma):
    tau   = torch.clamp(torch.as_tensor(tau,   device=S.device, dtype=S.dtype), min=1e-8)
    sigma = torch.clamp(torch.as_tensor(sigma, device=S.device, dtype=S.dtype), min=1e-8)
    d1 = (torch.log(S / K) + (r + 0.5*sigma**2)*tau) / (sigma*torch.sqrt(tau))
    d2 = d1 - sigma*torch.sqrt(tau)
    N = lambda x: 0.5*(1+torch.erf(x/math.sqrt(2.0)))
    return S*N(d1) - K*torch.exp(-r*tau)*N(d2)

def call_payoff(S_T, K): 
    return torch.clamp(S_T-K, min=0.0)

# -------------------- Losses --------------------
def loss_cvar(pnl, alpha):
    sorted_pnl, _ = torch.sort(pnl)
    k = max(1, int(alpha*len(sorted_pnl)))
    return -sorted_pnl[:k].mean()

def loss_entropic(pnl, gamma):
    x = -pnl / gamma
    m = torch.max(x.detach())
    return gamma*(torch.log(torch.mean(torch.exp(x-m))) + m)

# -------------------- Rollouts --------------------
def rollout_mlp(policy, S_paths, K, cost_rate, r, dt):
    N, T1 = S_paths.shape; T = T1-1
    cash = torch.zeros(N, device=S_paths.device, dtype=dtype)
    delta_prev = torch.zeros(N, device=S_paths.device, dtype=dtype)
    for k in range(T):
        S_t = S_paths[:,k]
        tau_norm = torch.full_like(S_t, (T-k)/T)
        delta_t = policy(S_t, tau_norm, delta_prev)
        trade_notional = (delta_t - delta_prev).abs()*S_t
        cash -= (delta_t - delta_prev)*S_t
        cash -= cost_rate*trade_notional
        if r != 0.0:
            cash *= torch.exp(torch.tensor(r*dt, device=S_paths.device, dtype=dtype))
        delta_prev = delta_t
    S_T = S_paths[:,-1]
    payoff = call_payoff(S_T, K)
    return cash + delta_prev*S_T - payoff

def rollout_gru(policy, S_paths, v_paths, K, cost_rate, r, dt):
    N, T1 = S_paths.shape; T = T1-1
    feats=[]
    for k in range(T):
        S_t=S_paths[:,k]
        sv_t=torch.sqrt(torch.clamp(v_paths[:,k], min=1e-8))
        tau_norm=torch.full_like(S_t,(T-k)/T)
        feats.append(torch.stack([S_t, sv_t, tau_norm], dim=-1))
    seq=torch.stack(feats, dim=1)  # [N,T,3]
    deltas=policy(seq)             # [N,T,2]
    cash=torch.zeros(N, device=S_paths.device, dtype=dtype)
    delta_prev=torch.zeros(N,2, device=S_paths.device, dtype=dtype)
    for k in range(T):
        S_t = S_paths[:,k]
        tau_k = torch.full_like(S_t, (T-k)*dt)
        sigma_k = torch.sqrt(torch.clamp(v_paths[:,k], min=1e-8))
        call_price_k = bs_call_price(S_t, K, tau_k, r, sigma_k)
        d_t = deltas[:,k,:]
        trade_stock = torch.abs(d_t[:,0]-delta_prev[:,0])*S_t
        trade_call  = torch.abs(d_t[:,1]-delta_prev[:,1])*call_price_k
        trade_costs = cost_rate*(trade_stock+trade_call)
        cash -= (d_t[:,0]-delta_prev[:,0])*S_t
        cash -= (d_t[:,1]-delta_prev[:,1])*call_price_k
        cash -= trade_costs
        if r != 0.0:
            cash *= torch.exp(torch.tensor(r*dt, device=S_paths.device, dtype=dtype))
        delta_prev = d_t
    S_T=S_paths[:,-1]; payoff=call_payoff(S_T,K)
    call_price_T=call_payoff(S_T,K)  # intrinsic at expiry
    return cash + delta_prev[:,0]*S_T + delta_prev[:,1]*call_price_T - payoff

# -------------------- Baseline (stock-only Δ) w/ costs --------------------
@torch.no_grad()
def rollout_bs_delta_stock_only(S_paths, K, r, dt, sigma_const=None, sigma_path=None, cost_rate=0.0):
    N, T1 = S_paths.shape; T=T1-1
    cash=torch.zeros(N, device=S_paths.device, dtype=dtype)
    delta_prev=torch.zeros(N, device=S_paths.device, dtype=dtype)
    for k in range(T):
        S_t=S_paths[:,k]
        tau=(T-k)*dt
        if sigma_path is None:
            sigma = sigma_const if sigma_const is not None else 0.2
        else:
            sigma = torch.sqrt(torch.clamp(sigma_path[:,k], min=1e-8))
        delta_t = bs_call_delta(S_t, K, tau, r, sigma)
        trade_notional = torch.abs(delta_t - delta_prev)*S_t
        cash -= (delta_t - delta_prev)*S_t
        cash -= cost_rate * trade_notional   # FAIR COST
        if r != 0.0:
            cash *= torch.exp(torch.tensor(r*dt, device=S_paths.device, dtype=dtype))
        delta_prev=delta_t
    S_T=S_paths[:,-1]; payoff=call_payoff(S_T,K)
    return cash + delta_prev*S_T - payoff

# -------------------- Replay helpers --------------------
@torch.no_grad()
def trace_mlp(policy, S_path, K, cost_rate, r, dt):
    N, T1 = S_path.shape; T=T1-1
    cash=torch.zeros(N, device=S_path.device, dtype=dtype)
    d_prev=torch.zeros(N, device=S_path.device, dtype=dtype)
    deltas=[]; trades=[]
    for k in range(T):
        S_t=S_path[:,k]
        tau_norm=torch.full_like(S_t,(T-k)/T)
        d_t=policy(S_t, tau_norm, d_prev)
        trade = (d_t - d_prev).abs()*S_t
        cash -= (d_t-d_prev)*S_t; cash -= cost_rate*trade
        if r!=0.0: cash*=torch.exp(torch.tensor(r*dt, device=S_path.device, dtype=dtype))
        deltas.append(d_t.item()); trades.append(trade.item()); d_prev=d_t
    pnl = (cash + d_prev*S_path[:,-1] - call_payoff(S_path[:,-1], K)).item()
    return deltas, trades, pnl

@torch.no_grad()
def trace_gru(policy, S_path, v_path, K, cost_rate, r, dt):
    N, T1=S_path.shape; T=T1-1
    feats=[]
    for k in range(T):
        S_t=S_path[:,k]; sv=torch.sqrt(torch.clamp(v_path[:,k], min=1e-8))
        tau=torch.full_like(S_t,(T-k)/T)
        feats.append(torch.stack([S_t, sv, tau], dim=-1))
    seq=torch.stack(feats, dim=1)
    deltas=policy(seq)  # [1,T,2]
    cash=torch.zeros(N, device=S_path.device, dtype=dtype)
    d_prev=torch.zeros(N,2, device=S_path.device, dtype=dtype)
    ds=[]; dc=[]; trades=[]
    for k in range(T):
        S_t=S_path[:,k]; tau_k=torch.full_like(S_t,(T-k)*dt)
        sig_k=torch.sqrt(torch.clamp(v_path[:,k], min=1e-8))
        call_p=bs_call_price(S_t, K, tau_k, r, sig_k)
        d_t=deltas[:,k,:]
        trade_val = torch.abs(d_t[:,0]-d_prev[:,0])*S_t + torch.abs(d_t[:,1]-d_prev[:,1])*call_p
        cash -= (d_t[:,0]-d_prev[:,0])*S_t
        cash -= (d_t[:,1]-d_prev[:,1])*call_p
        cash -= cost_rate*trade_val
        if r!=0.0: cash*=torch.exp(torch.tensor(r*dt, device=S_path.device, dtype=dtype))
        ds.append(d_t[0,0].item()); dc.append(d_t[0,1].item()); trades.append(trade_val.item()); d_prev=d_t
    pnl=(cash + d_prev[:,0]*S_path[:,-1] + d_prev[:,1]*call_payoff(S_path[:,-1],K) - call_payoff(S_path[:,-1],K)).item()
    return ds, dc, trades, pnl

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("Deep Hedging – Controls")

    st.subheader("Market & Liability")
    model_type = st.selectbox("Simulator", ["GBM (fast)", "Heston (stochastic vol)"])
    S0   = st.number_input("Spot S0", 50.0, 500.0, 100.0, step=1.0)
    K    = st.number_input("Strike K", 10.0, 500.0, 100.0, step=1.0)
    Tyrs = st.number_input("Maturity (years)", 0.25, 3.0, 1.0, step=0.25)
    steps = st.slider("Re-hedges (steps)", 8, 120, 40)
    dt = Tyrs / steps

    if model_type.startswith("GBM"):
        mu = st.slider("μ (drift)", -0.05, 0.15, 0.05, 0.005)
        sigma = st.slider("σ (vol)",   0.05, 0.80, 0.20, 0.01)
        r = st.slider("Risk-free r",  -0.02, 0.10, 0.00, 0.005)
    else:
        kappa = st.slider("κ (mean reversion)", 0.10, 4.00, 1.50, 0.05)
        theta = st.slider("θ (long-run var)",   0.01, 0.25, 0.04, 0.005)
        xi    = st.slider("ξ (vol-of-vol)",     0.05, 1.50, 0.30, 0.01)
        rho   = st.slider("ρ (corr)",          -0.95, 0.95, -0.60, 0.05)
        v0    = st.slider("v0 (initial var)",   0.01, 0.25, 0.04, 0.005)
        r     = st.slider("Risk-free r",      -0.02, 0.10, 0.00, 0.005)

    cost_rate = st.slider("Transaction cost λ", 0.000, 0.010, 0.001, 0.0005)

    st.subheader("Model & Risk")
    policy_kind = st.selectbox("Policy", ["MLP (stock-only)", "GRU (stock + call)"])
    loss_kind = st.selectbox("Risk objective", ["Entropic", "CVaR"])
    gamma = st.slider("γ (Entropic risk)", 0.000, 0.050, 0.010, 0.001)
    alpha = st.slider("α (CVaR tail %)",   0.01, 0.20, 0.05, 0.01)

    st.subheader("Training")
    epochs = st.slider("Epochs", 10, 800, 150)
    batch  = st.slider("Paths/epoch", 256, 10000, 4096, 256)
    testN  = st.slider("Test paths (OOS)", 1024, 50000, 16384, 512)
    lr     = st.slider("Learning rate", 1e-4, 5e-3, 1e-3, 1e-4, format="%.4f")
    seed   = st.number_input("Random seed", min_value=0, max_value=9999, value=42)

    st.subheader("Stress test knobs")
    shock_cost = st.slider("Cost multiplier", 0.5, 4.0, 2.0, 0.1)
    shock_vol  = st.slider("Vol multiplier",  0.5, 2.0, 1.3, 0.05)
    shock_rho  = st.slider("ρ (only Heston)", -0.95, 0.95, -0.90, 0.05)

# -------------------- Title / header --------------------
st.title("Deep Hedging Dashboard")
st.caption("Tail-risk minimization under costs • PyTorch + Streamlit UI.")
st.write(f"**Device:** `{device.type}`  •  **Steps:** {steps}  •  **dt:** {dt:0.4f}  •  **λ:** {cost_rate:0.3f}")

set_seeds(int(seed))

# -------------------- Train launcher (robust) --------------------
# Put this where your button currently is, BEFORE you build plots/tabs.

colA, colB = st.columns([1,1])
if "train_requested" not in st.session_state:
    st.session_state.train_requested = False

# Click the button -> set a flag; the flag survives reruns until we clear it.
if colA.button("Train / Re-run", type="primary", key="btn_train"):
    st.session_state.train_requested = True

compute_frontier_flag = colB.checkbox("Compute γ Frontier (quick)", value=False, key="chk_frontier")

# Create policy holder if missing
if "policy" not in st.session_state:
    st.session_state.policy = None
if "risk_hist" not in st.session_state:
    st.session_state.risk_hist = []

def _build_policy(policy_kind):
    if policy_kind.startswith("MLP"):
        return MLPHedger().to(device)
    else:
        return GRUHedger(hidden=128, num_instr=2).to(device)

# -------------------- Training happens here --------------------
if st.session_state.train_requested:
    # IMPORTANT: don't touch sidebar widgets after clicking — let this run.
    policy = _build_policy(policy_kind)
    opt = optim.Adam(policy.parameters(), lr=lr)
    risk_hist = []

    set_seeds(int(seed))
    t0 = time.time()

    with st.spinner("Training…"):
        for ep in range(1, epochs + 1):
            if model_type.startswith("GBM"):
                S = sim_gbm(batch, steps, S0, mu, sigma, dt)
                if policy_kind.startswith("MLP"):
                    pnl = rollout_mlp(policy, S, K, cost_rate, r, dt)
                else:
                    v_fake = torch.full_like(S, sigma**2)
                    pnl = rollout_gru(policy, S, v_fake, K, cost_rate, r, dt)
            else:
                S, v = sim_heston(batch, steps, S0, r, dt, kappa, theta, xi, rho, v0)
                pnl = rollout_mlp(policy, S, K, cost_rate, r, dt) if policy_kind.startswith("MLP") \
                      else rollout_gru(policy, S, v, K, cost_rate, r, dt)

            loss = loss_entropic(pnl, gamma) if loss_kind == "Entropic" else loss_cvar(pnl, alpha)
            opt.zero_grad(); loss.backward(); opt.step()
            risk_hist.append(float(loss.detach().cpu()))

            # show a lightweight heartbeat so you KNOW it's running
            if ep % max(epochs // 10, 1) == 0 or ep == 1:
                st.write(f"[Epoch {ep:>3}] risk={risk_hist[-1]:.4f}  mean={pnl.mean().item():+.3f}  std={pnl.std().item():.3f}")

    st.success(f"Training complete in {time.time()-t0:.1f}s")

    # store results and clear the trigger
    st.session_state.policy = policy
    st.session_state.risk_hist = risk_hist
    st.session_state.train_requested = False

# Short handles for the rest of your code:
policy = st.session_state.policy
risk_hist = st.session_state.risk_hist
# -------------------- Evaluation (OOS) --------------------
@torch.no_grad()
def evaluate(policy):
    if model_type.startswith("GBM"):
        S_test = sim_gbm(testN, steps, S0, mu, sigma, dt)
        if policy_kind.startswith("MLP"):
            pnl_pol  = rollout_mlp(policy, S_test, K, cost_rate, r, dt).cpu().numpy()
        else:
            v_fake   = torch.full_like(S_test, sigma**2)
            pnl_pol  = rollout_gru(policy, S_test, v_fake, K, cost_rate, r, dt).cpu().numpy()
        pnl_base = rollout_bs_delta_stock_only(S_test, K, r, dt,
                                               sigma_const=sigma, cost_rate=cost_rate).cpu().numpy()
    else:
        S_test, v_test = sim_heston(testN, steps, S0, r, dt, kappa, theta, xi, rho, v0)
        if policy_kind.startswith("MLP"):
            pnl_pol  = rollout_mlp(policy, S_test, K, cost_rate, r, dt).cpu().numpy()
        else:
            pnl_pol  = rollout_gru(policy, S_test, v_test, K, cost_rate, r, dt).cpu().numpy()
        pnl_base = rollout_bs_delta_stock_only(S_test, K, r, dt,
                                               sigma_path=v_test, cost_rate=cost_rate).cpu().numpy()
    return pnl_pol, pnl_base

pnl_pol, pnl_base = evaluate(policy) if policy is not None else (None, None)

# -------------------- Tabs --------------------
tab_res, tab_sum, tab_rep, tab_stress, tab_front = st.tabs(
    ["📊 Results", "🧠 Trader Summary", "🎥 Replay", "🧪 Stress Test", "⚖️ γ-Frontier"]
)

# ---------- RESULTS ----------
with tab_res:
    if pnl_pol is None:
        st.info("Click **Train / Re-run** to produce results.")
    else:
        mean_pol, std_pol, cvar_pol = float(np.mean(pnl_pol)), float(np.std(pnl_pol)), cvar_np(pnl_pol, 0.05)
        mean_bas, std_bas, cvar_bas = float(np.mean(pnl_base)), float(np.std(pnl_base)), cvar_np(pnl_base, 0.05)

        c1,c2,c3 = st.columns(3)
        c1.metric("Mean P&L (NN)", f"{mean_pol:.3f}")
        c2.metric("Std P&L (NN)",  f"{std_pol:.3f}")
        c3.metric("CVaR@5% (NN)",  f"{cvar_pol:.3f}")
        c1.metric("Mean P&L (Δ)",  f"{mean_bas:.3f}")
        c2.metric("Std P&L (Δ)",   f"{std_bas:.3f}")
        c3.metric("CVaR@5% (Δ)",   f"{cvar_bas:.3f}")

        fig1 = plt.figure(figsize=(6,4))
        plt.plot(risk_hist)
        plt.title("Risk Reduction During Training"); plt.xlabel("Epoch"); plt.ylabel("Objective"); plt.grid(True)
        st.pyplot(fig1)

        fig2 = plt.figure(figsize=(6,4))
        plt.hist(pnl_base, bins=80, alpha=0.55, label="Baseline: BS Δ (stock-only)")
        plt.hist(pnl_pol,  bins=80, alpha=0.55, label="Deep Hedging (policy)")
        plt.title("Out-of-Sample P&L Distribution"); plt.xlabel("Terminal P&L")
        plt.legend(); plt.grid(True)
        st.pyplot(fig2)
        st.caption("Fair costs applied to both. Lower CVaR/Std for the NN vs Δ ⇒ better tail control and smoother outcomes.")

# ---------- TRADER SUMMARY ----------
with tab_sum:
    if pnl_pol is None:
        st.info("Train the model first.")
    else:
        mean_pol, std_pol, cvar_pol = float(np.mean(pnl_pol)), float(np.std(pnl_pol)), cvar_np(pnl_pol, 0.05)
        mean_bas, std_bas, cvar_bas = float(np.mean(pnl_base)), float(np.std(pnl_base)), cvar_np(pnl_base, 0.05)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Policy Mean P&L", f"{mean_pol:.3f}", delta=f"{(mean_pol-mean_bas):+.3f} vs Δ")
        c2.metric("Policy Std",      f"{std_pol:.3f}",  delta=f"{(std_pol-std_bas):+.3f} vs Δ")
        c3.metric("Policy CVaR@5%",  f"{cvar_pol:.3f}", delta=f"{(cvar_pol-cvar_bas):+.3f} vs Δ")
        c4.metric("Tail Advantage (↓ better)", f"{(cvar_pol - cvar_bas):+.3f}")

        st.markdown("#### What this means (plain English)")
        bullets=[]
        bullets.append("Smaller **CVaR** ⇒ fewer severe-loss outcomes.")
        bullets.append("Lower **Std** ⇒ smoother P&L.")
        if mean_pol < mean_bas + 0.1:
            bullets.append("Slightly lower **mean P&L** is typical under risk-averse training (sacrifices some carry for safer tails).")
        if "Heston" in model_type:
            bullets.append("Heston (stoch-vol) makes Δ imperfect; the GRU can exploit vol-of-vol and reduce tails.")
        st.markdown("- " + "\n- ".join(bullets))

        res_df = pd.DataFrame({"pnl_policy": pnl_pol, "pnl_baseline": pnl_base})
        st.download_button("⬇️ Download P&L CSV", res_df.to_csv(index=False).encode(),
                           "results.csv", "text/csv")

# ---------- REPLAY ----------
with tab_rep:
    if policy is None:
        st.info("Train the model first.")
    else:
        st.write("One random path with hedge ratios and trade sizes.")
        if model_type.startswith("GBM"):
            S_one = sim_gbm(1, steps, S0, mu, sigma, dt)
            if policy_kind.startswith("MLP"):
                d, trades, pnl_one = trace_mlp(policy, S_one, K, cost_rate, r, dt)
                fig = plt.figure(figsize=(7,4))
                ax1=fig.add_subplot(211); ax2=fig.add_subplot(212, sharex=ax1)
                ax1.plot(S_one.squeeze().cpu().numpy(), label="Price"); ax1.legend(); ax1.grid(True)
                ax2.plot(d, label="δ_stock"); ax2.scatter(range(len(trades)), trades, s=10, label="Trade size")
                ax2.legend(); ax2.grid(True); ax2.set_xlabel("Step")
                st.pyplot(fig); st.caption(f"Replay P&L (one path): {pnl_one:.2f}")
            else:
                v_fake = torch.full_like(S_one, sigma**2)
                ds, dc, trades, pnl_one = trace_gru(policy, S_one, v_fake, K, cost_rate, r, dt)
                fig = plt.figure(figsize=(7,5))
                ax1=fig.add_subplot(311); ax2=fig.add_subplot(312, sharex=ax1); ax3=fig.add_subplot(313, sharex=ax1)
                ax1.plot(S_one.squeeze().cpu().numpy(), label="Price"); ax1.legend(); ax1.grid(True)
                ax2.plot(ds, label="δ_stock"); ax2.plot(dc, label="δ_call"); ax2.legend(); ax2.grid(True)
                ax3.scatter(range(len(trades)), trades, s=10, label="Trade cost proxy"); ax3.legend(); ax3.grid(True); ax3.set_xlabel("Step")
                st.pyplot(fig); st.caption(f"Replay P&L (one path): {pnl_one:.2f}")
        else:
            S_one, v_one = sim_heston(1, steps, S0, r, dt, kappa, theta, xi, rho, v0)
            if policy_kind.startswith("MLP"):
                d, trades, pnl_one = trace_mlp(policy, S_one, K, cost_rate, r, dt)
                fig = plt.figure(figsize=(7,4))
                ax1=fig.add_subplot(211); ax2=fig.add_subplot(212, sharex=ax1)
                ax1.plot(S_one.squeeze().cpu().numpy(), label="Price"); ax1.legend(); ax1.grid(True)
                ax2.plot(d, label="δ_stock"); ax2.scatter(range(len(trades)), trades, s=10, label="Trade size")
                ax2.legend(); ax2.grid(True); ax2.set_xlabel("Step")
                st.pyplot(fig); st.caption(f"Replay P&L (one path): {pnl_one:.2f}")
            else:
                ds, dc, trades, pnl_one = trace_gru(policy, S_one, v_one, K, cost_rate, r, dt)
                fig = plt.figure(figsize=(7,5))
                ax1=fig.add_subplot(311); ax2=fig.add_subplot(312, sharex=ax1); ax3=fig.add_subplot(313, sharex=ax1)
                ax1.plot(S_one.squeeze().cpu().numpy(), label="Price"); ax1.legend(); ax1.grid(True)
                ax2.plot(ds, label="δ_stock"); ax2.plot(dc, label="δ_call"); ax2.legend(); ax2.grid(True)
                ax3.scatter(range(len(trades)), trades, s=10, label="Trade cost proxy"); ax3.legend(); ax3.grid(True); ax3.set_xlabel("Step")
                st.pyplot(fig); st.caption(f"Replay P&L (one path): {pnl_one:.2f}")

# ---------- STRESS TEST ----------
with tab_stress:
    if policy is None:
        st.info("Train the model first.")
    else:
        with torch.no_grad():
            if model_type.startswith("GBM"):
                S_stress = sim_gbm(testN, steps, S0, mu, sigma*shock_vol, dt)
                v_stress = torch.full_like(S_stress, (sigma*shock_vol)**2)
                if policy_kind.startswith("MLP"):
                    pnl_pol_stress = rollout_mlp(policy, S_stress, K, cost_rate*shock_cost, r, dt).cpu().numpy()
                else:
                    pnl_pol_stress = rollout_gru(policy, S_stress, v_stress, K, cost_rate*shock_cost, r, dt).cpu().numpy()
                pnl_base_stress = rollout_bs_delta_stock_only(
                    S_stress, K, r, dt, sigma_const=sigma*shock_vol, cost_rate=cost_rate*shock_cost
                ).cpu().numpy()
            else:
                S_stress, v_stress = sim_heston(testN, steps, S0, r, dt,
                                                kappa, theta*(shock_vol**2/theta if theta>0 else 1.0),
                                                xi*shock_vol, shock_rho, v0*(shock_vol**2))
                if policy_kind.startswith("MLP"):
                    pnl_pol_stress = rollout_mlp(policy, S_stress, K, cost_rate*shock_cost, r, dt).cpu().numpy()
                else:
                    pnl_pol_stress = rollout_gru(policy, S_stress, v_stress, K, cost_rate*shock_cost, r, dt).cpu().numpy()
                pnl_base_stress = rollout_bs_delta_stock_only(
                    S_stress, K, r, dt, sigma_path=v_stress, cost_rate=cost_rate*shock_cost
                ).cpu().numpy()

        def metrics(p): return np.mean(p), np.std(p), cvar_np(p)
        m0,s0,c0 = metrics(pnl_pol);  m1,s1,c1 = metrics(pnl_pol_stress)
        col1,col2,col3 = st.columns(3)
        col1.metric("Δ Mean P&L (policy)", f"{m1-m0:+.2f}")
        col2.metric("Δ Std (policy)",      f"{s1-s0:+.2f}")
        col3.metric("Δ CVaR@5% (policy)",  f"{c1-c0:+.2f}")
        st.caption("Positive Δ CVaR ⇒ worse tails under stress. Robust policies keep Δ small.")

# ---------- γ-FRONTIER ----------
with tab_front:
    if not compute_frontier_flag:
        st.caption("Tick the checkbox near the Train button to compute the γ-frontier.")
    else:
        st.info("Training small models for a quick risk–return frontier …")
        gammas=[0.0025, 0.005, 0.01, 0.02]
        pts=[]
        for g in gammas:
            m = (MLPHedger().to(device) if policy_kind.startswith("MLP")
                 else GRUHedger(hidden=96, num_instr=2).to(device))
            optg = optim.Adam(m.parameters(), lr=1e-3)
            for _ in range(60):
                if model_type.startswith("GBM"):
                    S_b = sim_gbm(1024, steps, S0, mu, sigma, dt)
                    pnl_b = rollout_mlp(m, S_b, K, cost_rate, r, dt) if policy_kind.startswith("MLP") else \
                            rollout_gru(m, S_b, torch.full_like(S_b, sigma**2), K, cost_rate, r, dt)
                else:
                    S_b, v_b = sim_heston(1024, steps, S0, r, dt, kappa, theta, xi, rho, v0)
                    pnl_b = rollout_mlp(m, S_b, K, cost_rate, r, dt) if policy_kind.startswith("MLP") else \
                            rollout_gru(m, S_b, v_b, K, cost_rate, r, dt)
                loss_b = loss_entropic(pnl_b, g)
                optg.zero_grad(); loss_b.backward(); optg.step()
            with torch.no_grad():
                if model_type.startswith("GBM"):
                    S_o = sim_gbm(4096, steps, S0, mu, sigma, dt)
                    pnl_o = rollout_mlp(m, S_o, K, cost_rate, r, dt).cpu().numpy() if policy_kind.startswith("MLP") else \
                            rollout_gru(m, S_o, torch.full_like(S_o, sigma**2), K, cost_rate, r, dt).cpu().numpy()
                else:
                    S_o, v_o = sim_heston(4096, steps, S0, r, dt, kappa, theta, xi, rho, v0)
                    pnl_o = rollout_mlp(m, S_o, K, cost_rate, r, dt).cpu().numpy() if policy_kind.startswith("MLP") else \
                            rollout_gru(m, S_o, v_o, K, cost_rate, r, dt).cpu().numpy()
            pts.append((g, float(np.mean(pnl_o)), float(np.std(pnl_o)), cvar_np(pnl_o)))
        figf = plt.figure(figsize=(6,4))
        plt.scatter([c for *_,c in pts], [m for _,m,_,_ in pts])
        for (g,m,_,c) in pts:
            plt.annotate(f"γ={g}", (c,m), textcoords="offset points", xytext=(4,4), fontsize=8)
        plt.xlabel("CVaR@5% (lower better)"); plt.ylabel("Mean P&L (higher better)")
        plt.title("Risk–Return Frontier"); plt.grid(True)
        st.pyplot(figf)