# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
import torch

from deepmd.pt.model.descriptor.se_a import (
    DescrptSeA,
)
from deepmd.pt.model.model.ener import (
    DPModel,
)
from deepmd.pt.model.task.ener import (
    InvarFitting,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
    to_torch_tensor,
)

dtype = torch.float64


def finite_hessian(f, x, delta=1e-6):
    in_shape = x.shape
    assert len(in_shape) == 1
    y0 = f(x)
    out_shape = y0.shape
    res = np.empty(out_shape + in_shape + in_shape)
    for iidx in np.ndindex(*in_shape):
        for jidx in np.ndindex(*in_shape):
            i0 = np.zeros(in_shape)
            i1 = np.zeros(in_shape)
            i2 = np.zeros(in_shape)
            i3 = np.zeros(in_shape)
            i0[iidx] += delta
            i2[iidx] += delta
            i1[iidx] -= delta
            i3[iidx] -= delta
            i0[jidx] += delta
            i1[jidx] += delta
            i2[jidx] -= delta
            i3[jidx] -= delta
            y0 = f(x + i0)
            y1 = f(x + i1)
            y2 = f(x + i2)
            y3 = f(x + i3)
            res[(Ellipsis, *iidx, *jidx)] = (y0 + y3 - y1 - y2) / (4 * delta**2.0)
    return res


class HessianTest:
    def test(
        self,
    ):
        places = 8
        delta = 1e-4
        natoms = self.nloc
        nf = self.nf
        nv = self.nv
        cell0 = torch.rand([3, 3], dtype=dtype)
        cell0 = 1.0 * (cell0 + cell0.T) + 5.0 * torch.eye(3)
        cell1 = torch.rand([3, 3], dtype=dtype)
        cell1 = 1.0 * (cell1 + cell1.T) + 5.0 * torch.eye(3)
        cell = torch.stack([cell0, cell1])
        coord = torch.rand([nf, natoms, 3], dtype=dtype)
        coord = torch.matmul(coord, cell)
        cell = cell.view([nf, 9])
        coord = coord.view([nf, natoms * 3])
        atype = torch.stack(
            [
                torch.IntTensor([0, 0, 0, 1, 1]),
                torch.IntTensor([0, 1, 1, 0, 1]),
            ]
        ).view([nf, natoms])
        # assumes input to be numpy tensor
        # coord = coord.numpy()
        nfp, nap = 2, 3
        fparam = torch.rand([nf, nfp], dtype=dtype)
        aparam = torch.rand([nf, natoms * nap], dtype=dtype)

        coord = coord.view([nf, natoms, 3])
        coord = coord[:, [0, 1, 2, 3, 4], :]
        coord = coord.view([nf, natoms * 3])

        ret_dict0 = self.model_hess.forward_common(
            coord, atype, box=cell, fparam=fparam, aparam=aparam
        )
        ret_dict1 = self.model_valu.forward_common(
            coord, atype, box=cell, fparam=fparam, aparam=aparam
        )

        # print(ret_dict0["energy"].shape, ret_dict1["energy"].shape)
        torch.testing.assert_close(ret_dict0["energy"], ret_dict1["energy"])
        ana_hess = ret_dict0["energy_derv_r_derv_r"]

        # compute finite difference
        fnt_hess = []
        for ii in range(nf):

            def np_infer(
                xx,
            ):
                ret = self.model_valu.forward_common(
                    to_torch_tensor(xx).unsqueeze(0),
                    atype[ii].unsqueeze(0),
                    box=cell[ii].unsqueeze(0),
                    fparam=fparam[ii].unsqueeze(0),
                    aparam=aparam[ii].unsqueeze(0),
                )
                # detach
                ret = {kk: to_numpy_array(ret[kk]) for kk in ret}
                return ret

            def ff(xx):
                return np_infer(xx)["energy_redu"]

            xx = to_numpy_array(coord[ii])
            fnt_hess.append(finite_hessian(ff, xx, delta=delta).squeeze())

        fnt_hess = np.stack(fnt_hess).reshape([nf, nv, natoms * 3, natoms * 3])
        np.testing.assert_almost_equal(fnt_hess, to_numpy_array(ana_hess), decimal=6)


class TestDPModel(unittest.TestCase, HessianTest):
    def setUp(self):
        torch.manual_seed(2)
        self.nf = 2
        self.nloc = 5
        self.rcut = 4.0
        self.rcut_smth = 3.0
        self.sel = [15, 15]
        self.nt = 2
        self.nv = 1
        ds = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
            neuron=[2, 4, 8],
            axis_neuron=2,
        ).to(env.DEVICE)
        ft0 = InvarFitting(
            "energy",
            self.nt,
            ds.get_dim_out(),
            self.nv,
            distinguish_types=ds.distinguish_types(),
            do_hessian=True,
            neuron=[4, 4, 4],
        ).to(env.DEVICE)
        type_map = ["foo", "bar"]
        # TODO: dirty hack to avoid data stat!!!
        self.model_hess = DPModel(ds, ft0, type_map=type_map, resuming=True).to(
            env.DEVICE
        )
        self.model_valu = DPModel.deserialize(self.model_hess.serialize())

        # args = [to_torch_tensor(ii) for ii in [self.coord, self.atype, self.cell]]
        # ret0 = md0.forward_common(*args)

        # print(ret0["energy_redu"].shape)
        # print(ret0["energy_derv_r"].shape)
        # print(ret0["energy_derv_r_derv_r"].shape)

    def test_jit(self):
        torch.jit.script(self.model_hess)
        torch.jit.script(self.model_valu)
        # torch.jit.script(self.model_valu)
