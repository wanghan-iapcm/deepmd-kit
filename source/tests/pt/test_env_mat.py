# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
import torch

try:
    from deepmd.model_format import (
        EnvMat,
    )

    support_env_mat = True
except ModuleNotFoundError:
    support_env_mat = False
except ImportError:
    support_env_mat = False

from deepmd.pt.model.descriptor.env_mat import (
    prod_env_mat_se_a,
)
from deepmd.pt.utils import (
    env,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION


class TestCaseSingleFrameWithNlist:
    def setUp(self):
        # nloc == 3, nall == 4
        self.nloc = 3
        self.nall = 4
        self.nf, self.nt = 1, 2
        self.coord_ext = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, -2, 0],
            ],
            dtype=np.float64,
        ).reshape([1, self.nall * 3])
        self.atype_ext = np.array([0, 0, 1, 0], dtype=int).reshape([1, self.nall])
        # sel = [5, 2]
        self.sel = [5, 2]
        self.nlist = np.array(
            [
                [1, 3, -1, -1, -1, 2, -1],
                [0, -1, -1, -1, -1, 2, -1],
                [0, 1, -1, -1, -1, 0, -1],
            ],
            dtype=int,
        ).reshape([1, self.nloc, sum(self.sel)])
        self.rcut = 0.4
        self.rcut_smth = 2.2


# to be merged with the tf test case
@unittest.skipIf(not support_env_mat, "EnvMat not supported")
class TestEnvMat(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self):
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_consistency(
        self,
    ):
        rng = np.random.default_rng()
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)
        em0 = EnvMat(self.rcut, self.rcut_smth)
        mm0, ww0 = em0.call(self.coord_ext, self.atype_ext, self.nlist, davg, dstd)
        mm1, _, ww1 = prod_env_mat_se_a(
            torch.tensor(self.coord_ext, dtype=dtype),
            torch.tensor(self.nlist, dtype=int),
            torch.tensor(self.atype_ext[:, :nloc], dtype=int),
            davg,
            dstd,
            self.rcut,
            self.rcut_smth,
        )
        np.testing.assert_allclose(mm0, mm1)
        np.testing.assert_allclose(ww0, ww1)