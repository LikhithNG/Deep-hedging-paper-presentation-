# Deep Hedging: Neural Networks for Derivative Portfolio Management

## Overview
This repository contains a presentation on "Deep Hedging" by Buehler et al. (2018), which introduces a revolutionary framework for hedging derivatives using deep reinforcement learning instead of traditional mathematical models.

## Paper Details
- **Title**: Deep Hedging
- **Authors**: Hans Buehler, Lukas Gonon, Josef Teichmann, and Ben Wood
- **Published**: 2018
- **Link**: [arXiv:1802.03042](https://arxiv.org/abs/1802.03042)

## Key Concepts Covered

### 1. Problem Statement
- Traditional hedging relies on idealized "complete market" assumptions
- Real markets have frictions: transaction costs, liquidity constraints, bid-ask spreads
- Need for model-free approaches that learn from data

### 2. Deep Hedging Framework
- Uses semi-recurrent neural networks to approximate optimal trading strategies
- Maps market observables (prices, volatilities, past positions) to trading decisions
- Optimizes under various risk measures (CVaR, entropic risk)

### 3. Technical Architecture
```
Input: Market State (S_t, V_t) + Previous Position (δ_{t-1})
    ↓
Hidden Layers (ReLU activation)
    ↓
Output: Trading Decision (δ_t)
```

### 4. Key Results
- Recovers Black-Scholes hedging in frictionless markets
- Handles transaction costs with optimal scaling (ε^{2/3})
- Scales to high dimensions (tested up to 10 assets)
- Outperforms traditional Greeks-based approaches under market frictions

## Presentation Contents
- Introduction to hedging challenges
- Mathematical framework and neural network architecture
- Implementation details and optimization approach
- Numerical experiments (Heston model)
- Practical implications for risk management

## Applications
1. **Exotic derivative hedging** under realistic market conditions
2. **Portfolio optimization** with transaction costs
3. **Risk management** for large derivative books
4. **Alternative to Greeks** in incomplete markets

## Implementation Ideas
- Replace traditional delta-hedging with learned strategies
- Incorporate market microstructure into hedging decisions
- Handle path-dependent options and complex payoffs
- Adapt to changing market regimes without model recalibration

## Future Directions
- Extension to American options
- Incorporation of market impact models
- Multi-agent learning frameworks
- Real-time adaptation to market conditions

## References
```bibtex
@article{buehler2018deep,
  title={Deep hedging},
  author={Buehler, Hans and Gonon, Lukas and Teichmann, Josef and Wood, Ben},
  journal={arXiv preprint arXiv:1802.03042},
  year={2018}
}
```

## Presentation File
- `ppt final.pptx` - Main presentation slides

## Contact
For questions or discussions about this presentation, please open an issue in this repository.

---
