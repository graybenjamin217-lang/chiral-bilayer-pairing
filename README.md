
# Self-Confined Prime-Scattering Dynamics on $L^2(\mathbb{R})$ and Chiral Bilayer Pairing

[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/Code%20License-PolyForm%20Noncommercial%201.0.0-blue.svg)](LICENSE.md)
[![Manuscript License: CC BY-NC 4.0](https://img.shields.io/badge/Manuscript-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

Companion verification software and reproducible numerical benchmarks for **Paper 4** of the Primorial–Feedback Operator Series by **Benjamin Edward Gray**.

## Overview

Rewriting the variable-exponent feedback generator $f(x) = x^{1/x - 1}$ in logarithmic dilation coordinates $y = \ln x$ yields $f(e^y) = \exp(-V(y))$ with the smooth self-confining potential $V(y) = y(1 - e^{-y}) \ge 0$ on $L^2(\mathbb{R})$. This repository implements and verifies all finite-stage ($k \ge 1$) analytical theorems and numerical benchmarks in Paper 4:

1. **Theorem 2.1 (Active-Fiber Irreducibility):** Exact 1D common kernel on $V_{\chi_0}$ and full irreducibility of $\{\tilde{C}_p\}_{p \mid N_k}$ on the $(2^k - 1)$-dimensional active space $V_{\chi_0}^{\mathrm{act}}$.
2. **Theorem 3.1 & Proposition 3.2 (Self-Confined Continuum Operators):** Base bound states for $H_k^{(2)}$ and the two-component SUSY Dirac operator $D_k^{\mathrm{Dirac}}$ ($E_0 \approx 0.73063$, zero Witten index, $\pm E_n$ pairing).
3. **Section IV (Core vs. Global Quantum Chaos):** Resolution of Paper 2 Conjecture 7.2(2) through $k = 5$ ($d = 31$, converged to $M = 95$ modes), contrasting decaying prime bumps $f_p(y)$ against non-decaying envelopes ($\cos(y \ln p)$ and Feedback-Cos) via $\langle r \rangle$ plateaus, unfolded Spectral Form Factor rigidity, and periodic-orbit Fourier suppression.
4. **Theorem 5.1 (Topological Shear Zone):** Sign-flipping shear operator $\mathcal{D}_k^{\mathrm{shear}}$ with $M(y - y_*) = (y - y_*)(1 + e^{-(y - y_*)})$, verifying exact Jackiw–Rebbi index $2^k - 1$, $98.8\%$ mid-gap chiral polarization, and $\sqrt{k}$ commutator-drag scaling across $k = 2$–$6$.
5. **Theorem 5.2 (Stacked Toroidal Bilayer):** Bilayer $\mathcal{H}_k(\Delta y, \gamma)$ on $\mathbb{T}_{L_k}$, verifying exact bump-well gap capping ($\min|E| = 1.2009$ at $\Delta y = 0$, $\gamma = 1.2$), half-period glide-chiral symmetry ($\Delta y = \pm L_k/2$), and twist-induced shear pinch transitions ($\min|E| = 0.0033$ at $\Delta y = \pm 1.604$).

## Quick Start

```bash
pip install -r requirements.txt
pytest -v test_paper4.py
