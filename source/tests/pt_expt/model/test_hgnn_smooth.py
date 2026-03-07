# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import unittest

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
    nframes = coord.shape[0]
    if len(atype.shape) == 1:
        atype = atype.unsqueeze(0).expand(nframes, -1)
    coord_input = coord.to(dtype=dtype, device=env.DEVICE)
    cell_input = cell.reshape(nframes, 9).to(dtype=dtype, device=env.DEVICE)
    atype_input = atype.to(dtype=torch.long, device=env.DEVICE)
    coord_input.requires_grad_(True)
    result = model(coord_input, atype_input, cell_input)
    return result


class SmoothTest:
    """Test smoothness at cutoff boundaries.

    Places atoms near the edge (e_rcut) and angle (a_rcut) cutoffs,
    then applies small epsilon displacements. If the descriptor is smooth,
    the energy/force/virial should change by at most O(epsilon).
    """

    def test(self) -> None:
        generator = torch.Generator(device=env.DEVICE).manual_seed(GLOBAL_SEED)
        epsilon = self.epsilon
        aprec = self.aprec

        natoms = 10
        cell = 8.6 * torch.eye(3, dtype=dtype, device=env.DEVICE)
        atype0 = torch.arange(3, dtype=dtype, device=env.DEVICE)
        atype1 = torch.randint(
            0, 3, [natoms - 3], device=env.DEVICE, generator=generator
        )
        atype = torch.cat([atype0, atype1]).view([natoms])
        # Place atoms near both cutoffs:
        #   atom 0 at origin
        #   atom 1 at x ~ 4.0 (a_rcut boundary)
        #   atom 2 at y ~ 4.0 (a_rcut boundary)
        #   atom 3 at x ~ 6.0 (e_rcut boundary)
        #   atom 4 at y ~ 6.0 (e_rcut boundary)
        coord0 = torch.tensor(
            [
                0.0,
                0.0,
                0.0,
                4.0 - 0.5 * epsilon,
                0.0,
                0.0,
                0.0,
                4.0 - 0.5 * epsilon,
                0.0,
                6.0 - 0.5 * epsilon,
                0.0,
                0.0,
                0.0,
                6.0 - 0.5 * epsilon,
                0.0,
            ],
            dtype=dtype,
            device=env.DEVICE,
        ).view([-1, 3])
        coord1 = torch.rand(
            [natoms - coord0.shape[0], 3],
            dtype=dtype,
            device=env.DEVICE,
            generator=generator,
        )
        coord1 = torch.matmul(coord1, cell)
        coord = torch.concat([coord0, coord1], dim=0)

        coord0 = torch.clone(coord)
        coord1 = torch.clone(coord)
        coord1[1][0] += epsilon
        coord1[3][0] += epsilon
        coord2 = torch.clone(coord)
        coord2[2][1] += epsilon
        coord2[4][1] += epsilon
        coord3 = torch.clone(coord)
        coord3[1][0] += epsilon
        coord3[3][0] += epsilon
        coord3[2][1] += epsilon
        coord3[4][1] += epsilon

        test_keys = ["energy", "force", "virial"]

        result_0 = eval_model(self.model, coord0.unsqueeze(0), cell.unsqueeze(0), atype)
        ret0 = {key: result_0[key].squeeze(0) for key in test_keys}
        result_1 = eval_model(self.model, coord1.unsqueeze(0), cell.unsqueeze(0), atype)
        ret1 = {key: result_1[key].squeeze(0) for key in test_keys}
        result_2 = eval_model(self.model, coord2.unsqueeze(0), cell.unsqueeze(0), atype)
        ret2 = {key: result_2[key].squeeze(0) for key in test_keys}
        result_3 = eval_model(self.model, coord3.unsqueeze(0), cell.unsqueeze(0), atype)
        ret3 = {key: result_3[key].squeeze(0) for key in test_keys}

        def compare(ret0, ret1) -> None:
            for key in test_keys:
                if key in ["energy"]:
                    torch.testing.assert_close(ret0[key], ret1[key], rtol=0, atol=aprec)
                elif key in ["force"]:
                    torch.testing.assert_close(
                        1.0 + ret0[key], 1.0 + ret1[key], rtol=0, atol=aprec
                    )
                elif key == "virial":
                    torch.testing.assert_close(
                        1.0 + ret0[key], 1.0 + ret1[key], rtol=0, atol=aprec
                    )

        compare(ret0, ret1)
        compare(ret1, ret2)
        compare(ret0, ret3)


class TestEnergyModelHGNN(unittest.TestCase, SmoothTest):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_hgnn)
        self.model = get_model(model_params).to(env.DEVICE)
        self.model.eval()
        self.epsilon = 1e-5
        self.aprec = 1e-5


model_hgnn_enhanced = copy.deepcopy(model_hgnn)
model_hgnn_enhanced["descriptor"]["hgnn"]["use_hfea_v2e"] = True
model_hgnn_enhanced["descriptor"]["hgnn"]["use_cross_order_v2e"] = True


class TestEnergyModelHGNNEnhanced(unittest.TestCase, SmoothTest):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_hgnn_enhanced)
        self.model = get_model(model_params).to(env.DEVICE)
        self.model.eval()
        self.epsilon = 1e-5
        self.aprec = 1e-5


if __name__ == "__main__":
    unittest.main()
