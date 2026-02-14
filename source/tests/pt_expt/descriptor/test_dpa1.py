# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
import unittest

import numpy as np
import torch

from deepmd.dpmodel.descriptor.dpa1 import DescrptDPA1 as DPDescrptDPA1
from deepmd.pt_expt.descriptor.dpa1 import (
    DescrptDPA1,
)
from deepmd.pt_expt.utils import (
    env,
)
from deepmd.pt_expt.utils.env import (
    PRECISION_DICT,
)

from ...pt.model.test_env_mat import (
    TestCaseSingleFrameWithNlist,
)
from ...pt.model.test_mlp import (
    get_tols,
)
from ...seed import (
    GLOBAL_SEED,
)


class TestDescrptDPA1(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)
        self.device = env.DEVICE

    def test_consistency(self) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        _, _, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        for idt, sm, to, tm, prec, ect in itertools.product(
            [False, True],  # resnet_dt
            [False, True],  # smooth_type_embedding
            [False, True],  # type_one_side
            ["concat", "strip"],  # tebd_input_mode
            ["float64"],  # precision
            [False, True],  # use_econf_tebd
        ):
            dtype = PRECISION_DICT[prec]
            rtol, atol = get_tols(prec)
            err_msg = f"idt={idt} sm={sm} to={to} tm={tm} prec={prec} ect={ect}"

            dd0 = DescrptDPA1(
                self.rcut,
                self.rcut_smth,
                self.sel_mix,
                self.nt,
                attn_layer=2,
                precision=prec,
                resnet_dt=idt,
                smooth_type_embedding=sm,
                type_one_side=to,
                tebd_input_mode=tm,
                use_econf_tebd=ect,
                type_map=["O", "H"] if ect else None,
                seed=GLOBAL_SEED,
            ).to(self.device)
            dd0.se_atten.mean = torch.tensor(davg, dtype=dtype, device=self.device)
            dd0.se_atten.stddev = torch.tensor(dstd, dtype=dtype, device=self.device)
            rd0, _, _, _, _ = dd0(
                torch.tensor(self.coord_ext, dtype=dtype, device=self.device),
                torch.tensor(self.atype_ext, dtype=int, device=self.device),
                torch.tensor(self.nlist, dtype=int, device=self.device),
            )
            # serialization round-trip
            dd1 = DescrptDPA1.deserialize(dd0.serialize())
            rd1, _, _, _, _ = dd1(
                torch.tensor(self.coord_ext, dtype=dtype, device=self.device),
                torch.tensor(self.atype_ext, dtype=int, device=self.device),
                torch.tensor(self.nlist, dtype=int, device=self.device),
            )
            np.testing.assert_allclose(
                rd0.detach().cpu().numpy(),
                rd1.detach().cpu().numpy(),
                rtol=rtol,
                atol=atol,
                err_msg=err_msg,
            )
            # dp impl
            dd2 = DPDescrptDPA1.deserialize(dd0.serialize())
            rd2, _, _, _, _ = dd2.call(
                self.coord_ext,
                self.atype_ext,
                self.nlist,
            )
            np.testing.assert_allclose(
                rd0.detach().cpu().numpy(),
                rd2,
                rtol=rtol,
                atol=atol,
                err_msg=err_msg,
            )

    def test_exportable(self) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        _, _, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        for idt, prec in itertools.product(
            [False, True],
            ["float64", "float32"],
        ):
            dtype = PRECISION_DICT[prec]
            dd0 = DescrptDPA1(
                self.rcut,
                self.rcut_smth,
                self.sel_mix,
                self.nt,
                attn_layer=2,
                precision=prec,
                resnet_dt=idt,
                seed=GLOBAL_SEED,
            ).to(self.device)
            dd0.se_atten.mean = torch.tensor(davg, dtype=dtype, device=self.device)
            dd0.se_atten.stddev = torch.tensor(dstd, dtype=dtype, device=self.device)
            dd0 = dd0.eval()
            inputs = (
                torch.tensor(self.coord_ext, dtype=dtype, device=self.device),
                torch.tensor(self.atype_ext, dtype=int, device=self.device),
                torch.tensor(self.nlist, dtype=int, device=self.device),
            )
            torch.export.export(dd0, inputs)
