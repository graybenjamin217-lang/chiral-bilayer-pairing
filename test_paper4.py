"""
Automated verification suite for Paper 4 (5/5 tests pass in ~6s on CPU).
Run with: pytest -v test_paper4.py  OR  python -m unittest -v test_paper4.py
"""

import unittest
import numpy as np
from chiral_bilayer import (
    build_active_prime_channels,
    build_B1_hamiltonian,
    build_shear_operator,
    build_toroidal_bilayer,
    f_prime_bump,
    mean_spacing_ratio,
    solve_D_conf_base,
    solve_H0_basis,
)


class TestPaper4(unittest.TestCase):
    def test_theorem_2_1_active_fiber_irreducibility(self):
        expected_norms = {2: 1.770261, 3: 2.505158, 4: 3.205514, 5: 3.868214}
        for k, exp_norm in expected_norms.items():
            _, full_ch, act_ch = build_active_prime_channels(k, return_full=True)
            d = 2**k - 1
            for C_full, C_act in zip(full_ch, act_ch):
                self.assertLess(np.max(np.abs(C_full[-1, :])), 1e-14)
                self.assertLess(np.max(np.abs(C_full[:, -1])), 1e-14)
                self.assertEqual(C_act.shape, (d, d))
                self.assertLess(np.max(np.abs(C_act @ C_act @ C_act - C_act)), 1e-14)
            T_act = sum(act_ch)
            self.assertAlmostEqual(np.linalg.norm(T_act, 2), exp_norm, places=5)

    def test_theorem_3_1_base_continuum_operators(self):
        _, E0, _ = solve_H0_basis(M_modes=25)
        self.assertTrue(np.all(E0 > 0))
        self.assertTrue(np.all(np.diff(E0) > 0))
        ep, em = solve_D_conf_base(M_pairs=15)
        self.assertAlmostEqual(ep[0], 0.73063, places=3)
        self.assertLess(np.max(np.abs(ep - em)), 1e-4)

    def test_section_4_core_vs_global_chaos_k4(self):
        primes, chans = build_active_prime_channels(4)
        y_int, E0, psi = solve_H0_basis(y_max=54.0, N=2400, M_modes=60)
        env_bump = [lambda y, p=p: f_prime_bump(y, p) for p in primes]
        env_qp = [lambda y, p=p: np.cos(y * np.log(p)) for p in primes]
        ev_bump = np.linalg.eigvalsh(
            build_B1_hamiltonian(y_int, E0, psi, chans, env_bump, 4.0, break_tri=True)
        )
        ev_qp = np.linalg.eigvalsh(
            build_B1_hamiltonian(y_int, E0, psi, chans, env_qp, 4.0, break_tri=True)
        )
        r_bump_hi, _ = mean_spacing_ratio(ev_bump[(ev_bump >= 21.0) & (ev_bump < 29.0)])
        r_qp_hi, _ = mean_spacing_ratio(ev_qp[(ev_qp >= 21.0) & (ev_qp < 29.0)])
        self.assertLess(r_bump_hi, 0.45)
        self.assertGreater(r_qp_hi, 0.55)

    def test_theorem_5_1_jackiw_rebbi_index_and_polarization(self):
        k = 4
        d = 2**k - 1
        D0, _, _, _, _, _, _ = build_shear_operator(k, eps0=0.0, gamma=2.0, M_pairs=18)
        ev0 = np.linalg.eigvalsh(D0)
        zero_modes = int(np.sum(np.abs(ev0) < 1e-9))
        self.assertEqual(zero_modes, d)

        D1, _, _, _, _, up, _ = build_shear_operator(k, eps0=0.6, gamma=1.5, M_pairs=18)
        ev1, vec1 = np.linalg.eigh(D1)
        idx_mid = np.argsort(np.abs(ev1))[:d]
        n_u = up.shape[1] * d
        Z_mid = float(np.mean([np.sum(vec1[:n_u, i] ** 2) - np.sum(vec1[n_u:, i] ** 2) for i in idx_mid]))
        self.assertGreater(Z_mid, 0.95)

    def test_theorem_5_2_stacked_toroidal_bilayer(self):
        L4, H_lock = build_toroidal_bilayer(4, 0.0, 1.2)
        ev_lock = np.linalg.eigvalsh(H_lock)
        self.assertAlmostEqual(np.min(np.abs(ev_lock)), 1.2009, places=3)
        self.assertLess(np.max(np.abs(ev_lock + ev_lock[::-1])), 1e-11)

        _, H_half = build_toroidal_bilayer(4, L4 / 2.0, 1.2)
        ev_half = np.linalg.eigvalsh(H_half)
        self.assertLess(np.max(np.abs(ev_half + ev_half[::-1])), 1e-8)

        _, H_pinch = build_toroidal_bilayer(4, 1.604, 1.2)
        ev_pinch = np.linalg.eigvalsh(H_pinch)
        self.assertLess(np.min(np.abs(ev_pinch)), 0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
