"""
Core operator library for Paper 4:
Self-Confined Prime-Scattering Dynamics on L^2(R) and Chiral Bilayer Pairing.
"""

import math
import numpy as np
from scipy.linalg import eigh_tridiagonal

PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)


def f_pos(x):
    """Variable-exponent feedback generator f(x) = x^(1/x - 1), f(0) = 0."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    m = x > 1e-12
    xm = x[m]
    out[m] = np.exp(((1.0 - xm) / xm) * np.log(xm))
    return out


def V_pot(y):
    """Self-confining potential V(y) = y(1 - e^-y) >= 0."""
    return y * (1.0 - np.exp(-np.clip(y, -50.0, 50.0)))


def M_pot(y):
    """Monotone sign-flipping shear potential M(y) = y(1 + e^-y)."""
    return y * (1.0 + np.exp(-np.clip(y, -50.0, 50.0)))


def M_deriv(y):
    """Exact derivative M'(y) = 1 + (1 - y)e^-y > 0.864."""
    ey = np.exp(-np.clip(y, -50.0, 50.0))
    return 1.0 + (1.0 - y) * ey


def f_prime_bump(y, p):
    """Prime-localized envelope f_p(y) = exp(-V(y - ln p)) in (0, 1]."""
    u = y - np.log(p)
    v = u * (1.0 - np.exp(-np.clip(u, -20.0, 500.0)))
    return np.exp(-np.clip(v, 0.0, 700.0))


def f_prime_bump_periodic(y, p, L_k, n_images=4):
    """Smooth periodicized prime bump on T_{L_k}."""
    total = np.zeros_like(y, dtype=np.float64)
    for n in range(-n_images, n_images + 1):
        total += f_prime_bump(y + n * L_k, p)
    return total


def build_active_prime_channels(k, return_full=False):
    """Exact (2^k - 1) x (2^k - 1) active prime channels on V_{chi_0}^{act} (Theorem 2.1)."""
    primes = PRIMES[:k]
    P_tilde, R_tilde = [], []
    for q in primes:
        s = math.sqrt(q - 1.0)
        P_tilde.append(np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64))
        R_tilde.append(np.array([[1.0, -s], [-s, -1.0]], dtype=np.float64) / math.sqrt(q))
    full_channels, active_channels = [], []
    for j in range(k):
        M = np.array([[1.0]], dtype=np.float64)
        for i in range(k):
            M = np.kron(M, P_tilde[i] if i == j else R_tilde[i])
        full_channels.append(M)
        active_channels.append(M[:-1, :-1])
    if return_full:
        return primes, full_channels, active_channels
    return primes, active_channels


def mean_spacing_ratio(evals, tol=1e-11):
    """Unfolding-free consecutive level-spacing ratio <r>."""
    e = np.sort(np.asarray(evals, dtype=np.float64))
    s = np.diff(e)
    s = s[s > tol]
    if len(s) < 5:
        return float("nan"), 0
    r = np.minimum(s[:-1], s[1:]) / np.maximum(s[:-1], s[1:])
    return float(np.mean(r)), int(len(r))


def solve_H0_basis(y_min=-6.0, y_max=58.0, N=2600, M_modes=60):
    """Lowest M_modes bound states of H_0 = -d^2/dy^2 + V(y) on L^2(R)."""
    y = np.linspace(y_min, y_max, N)
    dy = y[1] - y[0]
    y_int = y[1:-1]
    diag = 2.0 / (dy**2) + V_pot(y_int)
    off = -np.ones(len(y_int) - 1) / (dy**2)
    evals, evecs = eigh_tridiagonal(diag, off, select="i", select_range=(0, M_modes - 1))
    return y_int, evals, evecs


def solve_D_conf_base(y_min=-4.5, y_max=22.0, N=1500, M_pairs=15):
    """Positive eigenvalues of the two SUSY partner blocks of (D_conf')^2."""
    y = np.linspace(y_min, y_max, N)
    dy = y[1] - y[0]
    y_int = y[1:-1]
    Vy = V_pot(y_int)
    Vp = 1.0 - (1.0 - y_int) * np.exp(-np.clip(y_int, -50.0, 50.0))
    off = -np.ones(len(y_int) - 1) / (dy**2)
    ep = eigh_tridiagonal(
        2.0 / (dy**2) + Vy**2 - Vp, off, select="i", select_range=(0, M_pairs - 1), eigvals_only=True
    )
    em = eigh_tridiagonal(
        2.0 / (dy**2) + Vy**2 + Vp, off, select="i", select_range=(0, M_pairs - 1), eigvals_only=True
    )
    return np.sqrt(np.maximum(ep, 0.0)), np.sqrt(np.maximum(em, 0.0))


def build_B1_hamiltonian(y_int, E0, psi, chans, env_funcs, coupling, break_tri=False):
    """Second-order self-confined Model B1 Hamiltonian H_k^(2) in the H_0 eigenbasis."""
    k = len(chans)
    d = chans[0].shape[0]
    dtype = np.complex128 if break_tri else np.float64
    H = np.kron(np.diag(E0), np.eye(d)).astype(dtype)
    Gs = [psi.T @ (env_funcs[j](y_int)[:, None] * psi) for j in range(k)]
    for j in range(k):
        H += coupling * np.kron(Gs[j], chans[j])
    if break_tri:
        for a in range(k):
            for b in range(a + 1, k):
                w_ab = env_funcs[a](y_int) * env_funcs[b](y_int)
                G_ab = psi.T @ (w_ab[:, None] * psi)
                comm = 1j * (chans[a] @ chans[b] - chans[b] @ chans[a])
                H += 0.5 * coupling * np.kron(G_ab, comm)
    return H


def solve_shear_basis(y_star, y_min=-4.5, y_max=22.0, N=1500, M_pairs=20):
    """Orthonormal Jackiw-Rebbi + bulk SUSY basis for Q_M = d/dy + M(y - y_star)."""
    y = np.linspace(y_min, y_max, N)
    dy = y[1] - y[0]
    y_int = y[1:-1]
    u = y_int - y_star
    Mw = M_pot(u)
    Mp = M_deriv(u)
    off = -np.ones(len(y_int) - 1) / (dy**2)
    ep, up = eigh_tridiagonal(2.0 / (dy**2) + Mw**2 - Mp, off, select="i", select_range=(0, M_pairs))
    em, um = eigh_tridiagonal(2.0 / (dy**2) + Mw**2 + Mp, off, select="i", select_range=(0, M_pairs - 1))
    E_bulk = np.sqrt(np.maximum(0.5 * (ep[1:] + em), 0.0))
    for n in range(M_pairs):
        u_vec = up[:, n + 1]
        Qu = np.gradient(u_vec, dy) + Mw * u_vec
        if np.dot(um[:, n], Qu) < 0:
            um[:, n] *= -1.0
    return y_int, E_bulk, up, um


def build_shear_operator(k, y_star=None, eps0=0.0, gamma=1.5, M_pairs=20):
    """Topological shear operator D_k^shear on L^2(R) x C^2 x V_{chi_0}^{act} (Theorem 5.1)."""
    primes, chans = build_active_prime_channels(k)
    d = chans[0].shape[0]
    if y_star is None:
        y_star = float(np.mean([math.log(p) for p in primes]))
    y_s, E_b, up, um = solve_shear_basis(y_star, M_pairs=M_pairs)
    bumps = [f_prime_bump(y_s, p) for p in primes]
    F_uu = [up.T @ (b[:, None] * up) for b in bumps]
    F_vv = [um.T @ (b[:, None] * um) for b in bumps]
    F_vu = [um.T @ (b[:, None] * up) for b in bumps]

    Q0 = np.zeros((M_pairs, M_pairs + 1), dtype=np.float64)
    for n in range(M_pairs):
        Q0[n, n + 1] = E_b[n]
    S0 = np.kron(Q0, np.eye(d))
    S_prime = sum(np.kron(F_vu[j], chans[j]) for j in range(k))
    S = S0 + gamma * S_prime
    Wp = eps0 * sum(np.kron(F_uu[j], chans[j]) for j in range(k))
    Wm = -eps0 * sum(np.kron(F_vv[j], chans[j]) for j in range(k))
    D = np.block([[Wp, S.T], [S, Wm]])
    return D, S, Wp, Wm, y_s, up, um


def build_toroidal_bilayer(k, dy_shift, gamma, eps0=2.5, M_fourier=12, N_quad=256):
    """Stacked chiral toroidal bilayer H_k(dy_shift, gamma) on T_{L_k} (Theorem 5.2)."""
    primes, chans = build_active_prime_channels(k)
    d = chans[0].shape[0]
    L_k = sum(math.log(p) for p in primes)
    m_vals = np.arange(-M_fourier, M_fourier + 1)
    n_m = len(m_vals)
    y_q = np.linspace(0.0, L_k, N_quad, endpoint=False)
    E_mat = np.exp(2j * np.pi * np.outer(m_vals, y_q) / L_k) / math.sqrt(N_quad)

    def build_A(shift):
        A = np.kron(np.diag(2.0 * np.pi * m_vals / L_k), np.eye(d)).astype(np.complex128)
        for j, p in enumerate(primes):
            w = f_prime_bump_periodic(y_q - shift, p, L_k)
            W_f = (E_mat * w[None, :]) @ E_mat.conj().T
            A += eps0 * np.kron(W_f, chans[j])
        return A

    A_plus = build_A(0.0)
    A_minus = build_A(dy_shift)
    I_tot = np.eye(n_m * d, dtype=np.complex128)
    H = np.block([[A_plus, gamma * I_tot], [gamma * I_tot, -A_minus]])
    return L_k, H
