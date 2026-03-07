# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import unittest

import numpy as np
import torch

from deepmd.pt_expt.model.get_model import (
    get_model,
)
from deepmd.pt_expt.utils import (
    env,
)

from ...seed import (
    GLOBAL_SEED,
)

dtype = torch.float64

model_hgnn = {
    "type_map": ["O", "H", "B"],
    "descriptor": {
        "type": "hgnn",
        "hgnn": {
            "n_dim": 20,
            "e_dim": 10,
            "a_dim": 8,
            "nlayers": 3,
            "e_rcut": 6.0,
            "e_rcut_smth": 3.0,
            "e_sel": 20,
            "a_rcut": 4.0,
            "a_rcut_smth": 2.0,
            "a_sel": 10,
            "axis_neuron": 4,
            "update_style": "res_residual",
            "update_residual": 0.1,
            "update_residual_init": "const",
        },
        "precision": "float64",
    },
    "fitting_net": {
        "neuron": [24, 24],
        "resnet_dt": True,
        "precision": "float64",
        "seed": 1,
    },
}


def eval_model(model, coord, cell, atype):
    """Evaluate the pt_expt EnergyModel.

    Parameters
    ----------
    model : EnergyModel
        The model to evaluate.
    coord : torch.Tensor
        Coordinates, shape [nf, natoms, 3].
    cell : torch.Tensor
        Cell, shape [nf, 3, 3].
    atype : torch.Tensor
        Atom types, shape [natoms] or [nf, natoms].

    Returns
    -------
    dict
        Model predictions with keys: energy, force, virial.
    """
    nframes = coord.shape[0]
    if len(atype.shape) == 1:
        atype = atype.unsqueeze(0).expand(nframes, -1)
    coord_input = coord.to(dtype=dtype, device=env.DEVICE)
    cell_input = cell.reshape(nframes, 9).to(dtype=dtype, device=env.DEVICE)
    atype_input = atype.to(dtype=torch.long, device=env.DEVICE)
    coord_input.requires_grad_(True)
    result = model(coord_input, atype_input, cell_input)
    return result


class TranslationTest:
    def test_translation(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(GLOBAL_SEED)
        natoms = 5
        cell = torch.rand([3, 3], dtype=dtype, device="cpu", generator=generator)
        cell = (cell + cell.T) + 5.0 * torch.eye(3, device="cpu")
        coord = torch.rand([natoms, 3], dtype=dtype, device="cpu", generator=generator)
        coord = torch.matmul(coord, cell)
        atype = torch.IntTensor([0, 0, 1, 1, 2])

        # random translation
        shift = torch.rand([1, 3], dtype=dtype, device="cpu", generator=generator)
        coord_shifted = coord + shift

        ret0 = eval_model(
            self.model,
            coord.unsqueeze(0),
            cell.unsqueeze(0),
            atype,
        )
        ret1 = eval_model(
            self.model,
            coord_shifted.unsqueeze(0),
            cell.unsqueeze(0),
            atype,
        )

        np.testing.assert_allclose(
            ret0["energy"].detach().cpu().numpy(),
            ret1["energy"].detach().cpu().numpy(),
            rtol=1e-7,
            atol=1e-7,
            err_msg="Energy not invariant under translation",
        )
        np.testing.assert_allclose(
            ret0["force"].detach().cpu().numpy(),
            ret1["force"].detach().cpu().numpy(),
            rtol=1e-7,
            atol=1e-7,
            err_msg="Force not invariant under translation",
        )


class RotationTest:
    def test_rotation(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(GLOBAL_SEED + 1)
        natoms = 5
        cell = torch.rand([3, 3], dtype=dtype, device="cpu", generator=generator)
        cell = (cell + cell.T) + 5.0 * torch.eye(3, device="cpu")
        coord = torch.rand([natoms, 3], dtype=dtype, device="cpu", generator=generator)
        coord = torch.matmul(coord, cell)
        atype = torch.IntTensor([0, 0, 1, 1, 2])

        # random rotation matrix via QR
        rand_mat = torch.rand([3, 3], dtype=dtype, device="cpu", generator=generator)
        rot_mat, _ = torch.linalg.qr(rand_mat)
        # ensure proper rotation (det = +1)
        if torch.det(rot_mat) < 0:
            rot_mat[:, 0] = -rot_mat[:, 0]

        coord_rot = torch.matmul(coord, rot_mat)
        cell_rot = torch.matmul(cell, rot_mat)

        ret0 = eval_model(
            self.model,
            coord.unsqueeze(0),
            cell.unsqueeze(0),
            atype,
        )
        ret1 = eval_model(
            self.model,
            coord_rot.unsqueeze(0),
            cell_rot.unsqueeze(0),
            atype,
        )

        np.testing.assert_allclose(
            ret0["energy"].detach().cpu().numpy(),
            ret1["energy"].detach().cpu().numpy(),
            rtol=1e-7,
            atol=1e-7,
            err_msg="Energy not invariant under rotation",
        )
        # force covariance: F_rot = F @ R
        force0 = ret0["force"].detach().cpu().numpy()
        force1 = ret1["force"].detach().cpu().numpy()
        force0_rot = force0 @ rot_mat.numpy()
        np.testing.assert_allclose(
            force0_rot,
            force1,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Force not covariant under rotation",
        )


class PermutationTest:
    def test_permutation(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(GLOBAL_SEED + 2)
        natoms = 5
        cell = torch.rand([3, 3], dtype=dtype, device="cpu", generator=generator)
        cell = (cell + cell.T) + 5.0 * torch.eye(3, device="cpu")
        coord = torch.rand([natoms, 3], dtype=dtype, device="cpu", generator=generator)
        coord = torch.matmul(coord, cell)
        atype = torch.IntTensor([0, 0, 1, 1, 2])

        # random permutation
        perm = torch.randperm(natoms, generator=generator)
        coord_perm = coord[perm]
        atype_perm = atype[perm]

        ret0 = eval_model(
            self.model,
            coord.unsqueeze(0),
            cell.unsqueeze(0),
            atype,
        )
        ret1 = eval_model(
            self.model,
            coord_perm.unsqueeze(0),
            cell.unsqueeze(0),
            atype_perm,
        )

        np.testing.assert_allclose(
            ret0["energy"].detach().cpu().numpy(),
            ret1["energy"].detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
            err_msg="Energy not invariant under permutation",
        )
        # force permutation: F_perm[i] = F[perm[i]]
        force0 = ret0["force"].detach().cpu().numpy()
        force1 = ret1["force"].detach().cpu().numpy()
        np.testing.assert_allclose(
            force0[0][perm.numpy()],
            force1[0],
            rtol=1e-10,
            atol=1e-10,
            err_msg="Force not equivariant under permutation",
        )


class TestEnergyModelHGNN(
    unittest.TestCase, TranslationTest, RotationTest, PermutationTest
):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_hgnn)
        self.model = get_model(model_params).to(env.DEVICE)
        self.model.eval()


model_hgnn_enhanced = copy.deepcopy(model_hgnn)
model_hgnn_enhanced["descriptor"]["hgnn"]["use_hfea_v2e"] = True
model_hgnn_enhanced["descriptor"]["hgnn"]["use_cross_order_v2e"] = True


class TestEnergyModelHGNNEnhanced(
    unittest.TestCase, TranslationTest, RotationTest, PermutationTest
):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_hgnn_enhanced)
        self.model = get_model(model_params).to(env.DEVICE)
        self.model.eval()


if __name__ == "__main__":
    unittest.main()
