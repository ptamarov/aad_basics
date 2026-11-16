from unittest import TestCase, main
import numpy as np
from numpy.random import default_rng

from svi_ift import jacobian_svi
from svi import SliceSVI


class TestSVIJacobian(TestCase):

    def test_against_bump(self):

        zs = np.linspace(-3, 3, 16)
        rng = default_rng(1)
        num_samples = 1000

        a = 0.05 + np.sqrt(0.025) * rng.normal(size=num_samples)
        b = 0.05 + np.sqrt(0.025) * rng.normal(size=num_samples)
        r = np.clip(np.sqrt(0.50) * rng.normal(size=num_samples), a_max=1, a_min=-1)
        m = np.sqrt(0.05) * rng.normal(size=num_samples)
        s = 1.00 * np.exp(np.sqrt(0.025) * rng.normal(size=num_samples) - 0.025)

        for i in range(num_samples):
            P = np.asarray([a[i], b[i], r[i], m[i], s[i]])

            slice = SliceSVI(1.00)
            slice.params = P

            J = jacobian_svi(zs, P)

            # Compute derivatives using discrete approximation

            bump = 1e-5

            K = np.ones_like(J)
            for i in range(len(P)):
                tmp = P.copy()
                tmp[i] += bump
                new = SliceSVI(1.00)
                new.params = tmp
                base_up = new.bstermvar(zs)
                tmp = P.copy()
                tmp[i] -= bump
                new = SliceSVI(1.00)
                new.params = tmp
                base_down = new.bstermvar(zs)
                K[:, i] = (base_up - base_down) / (2 * bump)

            assert np.all(np.isclose(K, J))


if __name__ == "__main__":
    main()
