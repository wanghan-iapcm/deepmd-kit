# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import pytest
import torch

from deepmd.dpmodel.descriptor.se_atten_v2 import DescrptSeAttenV2 as DPDescrptSeAttenV2
from deepmd.pt_expt.descriptor.se_atten_v2 import (
    DescrptSeAttenV2,
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


class TestDescrptSeAttenV2(TestCaseSingleFrameWithNlist):
    def setup_method(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)
        self.device = env.DEVICE

    @pytest.mark.parametrize("idt", [False, True])  # resnet_dt
    @pytest.mark.parametrize("to", [False, True])  # type_one_side
    @pytest.mark.parametrize("prec", ["float64"])  # precision
    @pytest.mark.parametrize("ect", [False, True])  # use_econf_tebd
    def test_consistency(self, idt, to, prec, ect) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        _, _, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        dtype = PRECISION_DICT[prec]
        rtol, atol = get_tols(prec)
        err_msg = f"idt={idt} to={to} prec={prec} ect={ect}"

        dd0 = DescrptSeAttenV2(
            self.rcut,
            self.rcut_smth,
            self.sel_mix,
            self.nt,
            attn_layer=2,
            precision=prec,
            resnet_dt=idt,
            type_one_side=to,
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
        dd1 = DescrptSeAttenV2.deserialize(dd0.serialize())
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
        dd2 = DPDescrptSeAttenV2.deserialize(dd0.serialize())
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

    @pytest.mark.parametrize("idt", [False, True])  # resnet_dt
    @pytest.mark.parametrize("prec", ["float64", "float32"])  # precision
    def test_exportable(self, idt, prec) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        _, _, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        dtype = PRECISION_DICT[prec]
        dd0 = DescrptSeAttenV2(
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
