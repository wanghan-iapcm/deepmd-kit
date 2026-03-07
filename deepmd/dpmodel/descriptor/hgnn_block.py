# SPDX-License-Identifier: LGPL-3.0-or-later
from collections.abc import (
    Callable,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel import (
    PRECISION_DICT,
    NativeOP,
)
from deepmd.dpmodel.array_api import (
    Array,
    xp_take_along_axis,
)
from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.utils import (
    EnvMat,
    PairExcludeMask,
)
from deepmd.dpmodel.utils.env_mat_stat import (
    EnvMatStatSe,
)
from deepmd.dpmodel.utils.network import (
    NativeLayer,
    get_activation_fn,
)
from deepmd.dpmodel.utils.safe_gradient import (
    safe_for_vector_norm,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.utils.env_mat_stat import (
    StatItem,
)
from deepmd.utils.path import (
    DPPath,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .descriptor import (
    DescriptorBlock,
)
from .repformers import (
    _cal_hg,
    _make_nei_g1,
    get_residual,
    symmetrization_op,
)


class HGNNLayer(NativeOP):
    """Single HGNN message passing layer.

    Implements V→E updates for order-2 (bond) and order-3 (angle) hyperedges,
    E→V aggregation back to nodes, and symmetrization for invariant output.

    Parameters
    ----------
    e_rcut : float
        Edge cutoff radius.
    e_rcut_smth : float
        Edge smooth cutoff radius.
    e_sel : int or list[int]
        Edge neighbor selection.
    a_rcut : float
        Angle cutoff radius.
    a_rcut_smth : float
        Angle smooth cutoff radius.
    a_sel : int
        Angle neighbor selection.
    ntypes : int
        Number of atom types.
    n_dim : int
        Node feature dimension.
    e_dim : int
        Edge (order-2) feature dimension.
    a_dim : int
        Angle (order-3) feature dimension.
    axis_neuron : int
        Number of axis neurons for symmetrization.
    activation_function : str
        Activation function name.
    update_style : str
        Residual update style.
    update_residual : float
        Residual weight scale.
    update_residual_init : str
        Residual weight initialization mode.
    precision : str
        Floating point precision.
    seed : int or list[int] or None
        Random seed.
    trainable : bool
        Whether parameters are trainable.
    """

    def __init__(
        self,
        e_rcut: float,
        e_rcut_smth: float,
        e_sel: int | list[int],
        a_rcut: float,
        a_rcut_smth: float,
        a_sel: int,
        ntypes: int,
        n_dim: int = 128,
        e_dim: int = 64,
        a_dim: int = 64,
        axis_neuron: int = 4,
        activation_function: str = "silu",
        update_style: str = "res_residual",
        update_residual: float = 0.1,
        update_residual_init: str = "const",
        precision: str = "float64",
        seed: int | list[int] | None = None,
        trainable: bool = True,
        use_hfea_v2e: bool = False,
        use_cross_order_v2e: bool = False,
    ) -> None:
        super().__init__()
        self.epsilon = 1e-4
        self.e_rcut = float(e_rcut)
        self.e_rcut_smth = float(e_rcut_smth)
        self.ntypes = ntypes
        e_sel = [e_sel] if isinstance(e_sel, int) else e_sel
        self.nnei = sum(e_sel)
        assert len(e_sel) == 1
        self.e_sel = e_sel
        self.a_rcut = a_rcut
        self.a_rcut_smth = a_rcut_smth
        self.a_sel = a_sel
        self.n_dim = n_dim
        self.e_dim = e_dim
        self.a_dim = a_dim
        self.axis_neuron = axis_neuron
        self.activation_function = activation_function
        self.act = get_activation_fn(self.activation_function)
        self.update_style = update_style
        self.update_residual = update_residual
        self.update_residual_init = update_residual_init
        self.precision = precision
        self.seed = seed
        self.prec = PRECISION_DICT[precision]
        self.use_hfea_v2e = use_hfea_v2e
        self.use_cross_order_v2e = use_cross_order_v2e

        assert update_residual_init in [
            "norm",
            "const",
        ], "'update_residual_init' only support 'norm' or 'const'!"

        self.n_residual = []
        self.e_residual = []
        self.a_residual = []

        # -- V→E order-2: phi_2(x_nei, d) -> rho_2(x_center, phi_2_out[, e_prev]) --
        self.phi_2 = NativeLayer(
            n_dim + 1,
            e_dim,
            precision=precision,
            seed=child_seed(seed, 0),
            trainable=trainable,
        )
        rho_2_in = n_dim + e_dim + (e_dim if use_hfea_v2e else 0)
        self.rho_2 = NativeLayer(
            rho_2_in,
            e_dim,
            precision=precision,
            seed=child_seed(seed, 1),
            trainable=trainable,
        )
        if self.update_style == "res_residual":
            self.e_residual.append(
                get_residual(
                    e_dim,
                    self.update_residual,
                    self.update_residual_init,
                    precision=precision,
                    seed=child_seed(seed, 2),
                    trainable=trainable,
                )
            )

        # -- V→E order-3: phi_3(x_nei, d[, e]) -> rho_3(x_center, sum, cosine[, a_prev]) --
        phi_3_in = n_dim + 1 + (e_dim if use_cross_order_v2e else 0)
        self.phi_3 = NativeLayer(
            phi_3_in,
            a_dim,
            precision=precision,
            seed=child_seed(seed, 3),
            trainable=trainable,
        )
        rho_3_in = n_dim + a_dim + 1 + (a_dim if use_hfea_v2e else 0)
        self.rho_3 = NativeLayer(
            rho_3_in,
            a_dim,
            precision=precision,
            seed=child_seed(seed, 4),
            trainable=trainable,
        )
        if self.update_style == "res_residual":
            self.a_residual.append(
                get_residual(
                    a_dim,
                    self.update_residual,
                    self.update_residual_init,
                    precision=precision,
                    seed=child_seed(seed, 5),
                    trainable=trainable,
                )
            )

        # -- E→V order-2: psi_2 aggregates edge features --
        self.psi_2 = NativeLayer(
            e_dim,
            n_dim,
            precision=precision,
            seed=child_seed(seed, 6),
            trainable=trainable,
        )
        if self.update_style == "res_residual":
            self.n_residual.append(
                get_residual(
                    n_dim,
                    self.update_residual,
                    self.update_residual_init,
                    precision=precision,
                    seed=child_seed(seed, 7),
                    trainable=trainable,
                )
            )

        # -- E→V order-3: psi_3 aggregates angle features --
        self.psi_3 = NativeLayer(
            a_dim,
            n_dim,
            precision=precision,
            seed=child_seed(seed, 8),
            trainable=trainable,
        )
        if self.update_style == "res_residual":
            self.n_residual.append(
                get_residual(
                    n_dim,
                    self.update_residual,
                    self.update_residual_init,
                    precision=precision,
                    seed=child_seed(seed, 9),
                    trainable=trainable,
                )
            )

        # -- Node self-update: psi_self --
        self.psi_self = NativeLayer(
            n_dim,
            n_dim,
            precision=precision,
            seed=child_seed(seed, 10),
            trainable=trainable,
        )
        if self.update_style == "res_residual":
            self.n_residual.append(
                get_residual(
                    n_dim,
                    self.update_residual,
                    self.update_residual_init,
                    precision=precision,
                    seed=child_seed(seed, 11),
                    trainable=trainable,
                )
            )

        # -- Symmetrization: psi_sym --
        self.n_sym_dim = (n_dim + e_dim) * axis_neuron
        self.psi_sym = NativeLayer(
            self.n_sym_dim,
            n_dim,
            precision=precision,
            seed=child_seed(seed, 12),
            trainable=trainable,
        )
        if self.update_style == "res_residual":
            self.n_residual.append(
                get_residual(
                    n_dim,
                    self.update_residual,
                    self.update_residual_init,
                    precision=precision,
                    seed=child_seed(seed, 13),
                    trainable=trainable,
                )
            )

    def call(
        self,
        node_ebd_ext: Array,  # nf x nall x n_dim
        edge_ebd: Array,  # nf x nloc x nnei x e_dim
        h2: Array,  # nf x nloc x nnei x 3
        angle_ebd: Array,  # nf x nloc x a_sel x a_sel x a_dim
        nlist: Array,  # nf x nloc x nnei
        nlist_mask: Array,  # nf x nloc x nnei
        sw: Array,  # nf x nloc x nnei
        a_nlist_mask: Array,  # nf x nloc x a_sel
        a_sw: Array,  # nf x nloc x a_sel
        edge_input: Array,  # nf x nloc x nnei x 1  (1/r or r)
        a_edge_input: Array,  # nf x nloc x a_sel x 1  (1/r for angle neis)
        cosine_ij: Array,  # nf x nloc x a_sel x a_sel  (cosine matrix)
    ) -> tuple[Array, Array, Array]:
        """Single HGNN layer update.

        Returns
        -------
        n_updated : nf x nloc x n_dim
        e_updated : nf x nloc x nnei x e_dim
        a_updated : nf x nloc x a_sel x a_sel x a_dim
        """
        xp = array_api_compat.array_namespace(
            node_ebd_ext,
            edge_ebd,
            h2,
            angle_ebd,
            nlist,
            nlist_mask,
            sw,
            a_nlist_mask,
            a_sw,
        )
        nb, nloc, nnei = nlist.shape
        node_ebd = node_ebd_ext[:, :nloc, :]

        # -- Gather neighbor node features --
        # nf x nloc x nnei x n_dim
        nei_node_ebd = _make_nei_g1(node_ebd_ext, nlist)

        n_update_list: list[Array] = [node_ebd]
        e_update_list: list[Array] = [edge_ebd]
        a_update_list: list[Array] = [angle_ebd]

        # ============================================================
        # V→E order-2 update
        # ============================================================
        # phi_2: process peripheral atom + distance
        # nf x nloc x nnei x (n_dim + 1)
        phi2_input = xp.concat([nei_node_ebd, edge_input], axis=-1)
        # nf x nloc x nnei x e_dim
        periph_msg = self.act(self.phi_2(phi2_input))

        # rho_2: combine center + peripheral message [+ existing edge feature]
        # nf x nloc x 1 x n_dim -> nf x nloc x nnei x n_dim
        node_tiled = xp.broadcast_to(
            node_ebd[:, :, xp.newaxis, :], (*periph_msg.shape[:-1], self.n_dim)
        )
        rho2_parts = [node_tiled, periph_msg]
        if self.use_hfea_v2e:
            rho2_parts.append(edge_ebd)
        rho2_input = xp.concat(rho2_parts, axis=-1)
        # nf x nloc x nnei x e_dim
        he2_update = self.act(self.rho_2(rho2_input))
        e_update_list.append(he2_update)

        # ============================================================
        # V→E order-3 update
        # ============================================================
        # phi_3: process each peripheral + distance (for angle neighbors only)
        # nf x nloc x a_sel x n_dim
        a_nei_node_ebd = nei_node_ebd[:, :, : self.a_sel, :]
        # Apply angle mask to neighbor embeddings
        a_nei_node_ebd = xp.where(
            xp.expand_dims(a_nlist_mask, axis=-1),
            a_nei_node_ebd,
            xp.zeros_like(a_nei_node_ebd),
        )
        phi3_parts = [a_nei_node_ebd, a_edge_input]
        if self.use_cross_order_v2e:
            # include edge features of each arm (center→peripheral)
            a_edge_ebd = edge_ebd[:, :, : self.a_sel, :]
            a_edge_ebd = xp.where(
                xp.expand_dims(a_nlist_mask, axis=-1),
                a_edge_ebd,
                xp.zeros_like(a_edge_ebd),
            )
            phi3_parts.append(a_edge_ebd)
        phi3_input = xp.concat(phi3_parts, axis=-1)
        # nf x nloc x a_sel x a_dim
        periph_phi3 = self.act(self.phi_3(phi3_input))

        # DeepSets sum: periph_i + periph_k for pair (i,k)
        # nf x nloc x a_sel x 1 x a_dim + nf x nloc x 1 x a_sel x a_dim
        # -> nf x nloc x a_sel x a_sel x a_dim
        periph_sum = (
            periph_phi3[:, :, :, xp.newaxis, :] + periph_phi3[:, :, xp.newaxis, :, :]
        )

        # rho_3: center + summed peripherals + cosine angle
        # nf x nloc x 1 x 1 x n_dim
        node_for_angle = node_ebd[:, :, xp.newaxis, xp.newaxis, :]
        # broadcast to nf x nloc x a_sel x a_sel x n_dim
        node_for_angle = xp.broadcast_to(
            node_for_angle,
            (*periph_sum.shape[:-1], self.n_dim),
        )
        # nf x nloc x a_sel x a_sel x 1
        cos_input = cosine_ij[:, :, :, :, xp.newaxis] / (xp.pi**0.5)

        rho3_parts = [node_for_angle, periph_sum, cos_input]
        if self.use_hfea_v2e:
            rho3_parts.append(angle_ebd)
        rho3_input = xp.concat(rho3_parts, axis=-1)
        # nf x nloc x a_sel x a_sel x a_dim
        he3_update = self.act(self.rho_3(rho3_input))
        a_update_list.append(he3_update)

        # ============================================================
        # E→V aggregation
        # ============================================================
        # From order-2: aggregate edge features centered at v
        # nf x nloc x nnei x e_dim * sw -> sum -> nf x nloc x e_dim
        e2_weighted = edge_ebd * xp.expand_dims(sw, axis=-1)
        e2_agg = xp.sum(e2_weighted, axis=2) / (float(self.nnei) ** 0.5)
        node_from_e2 = self.act(self.psi_2(e2_agg))
        n_update_list.append(node_from_e2)

        # From order-3: aggregate angle features centered at v
        # nf x nloc x a_sel x a_sel x a_dim
        a_sw_2d = (
            a_sw[:, :, :, xp.newaxis, xp.newaxis]
            * a_sw[:, :, xp.newaxis, :, xp.newaxis]
        )
        a3_weighted = angle_ebd * a_sw_2d
        # sum over both angle neighbor dims
        a3_agg = xp.sum(xp.sum(a3_weighted, axis=3), axis=2) / float(self.a_sel)
        node_from_e3 = self.act(self.psi_3(a3_agg))
        n_update_list.append(node_from_e3)

        # Node self-update
        node_self = self.act(self.psi_self(node_ebd))
        n_update_list.append(node_self)

        # Symmetrization: grrg from edge_ebd and nei_node_ebd
        sym_list: list[Array] = []
        sym_list.append(
            symmetrization_op(edge_ebd, h2, nlist_mask, sw, self.axis_neuron)
        )
        sym_list.append(
            symmetrization_op(nei_node_ebd, h2, nlist_mask, sw, self.axis_neuron)
        )
        node_sym = self.act(self.psi_sym(xp.concat(sym_list, axis=-1)))
        n_update_list.append(node_sym)

        # ============================================================
        # Residual updates
        # ============================================================
        n_updated = self.list_update(n_update_list, "node")
        e_updated = self.list_update(e_update_list, "edge")
        a_updated = self.list_update(a_update_list, "angle")

        return n_updated, e_updated, a_updated

    def list_update_res_avg(self, update_list: list[Array]) -> Array:
        nitem = len(update_list)
        uu = update_list[0]
        for ii in range(1, nitem):
            uu = uu + update_list[ii]
        return uu / (float(nitem) ** 0.5)

    def list_update_res_incr(self, update_list: list[Array]) -> Array:
        nitem = len(update_list)
        uu = update_list[0]
        scale = 1.0 / (float(nitem - 1) ** 0.5) if nitem > 1 else 0.0
        for ii in range(1, nitem):
            uu = uu + scale * update_list[ii]
        return uu

    def list_update_res_residual(
        self, update_list: list[Array], update_name: str = "node"
    ) -> Array:
        uu = update_list[0]
        if update_name == "node":
            for ii, vv in enumerate(self.n_residual):
                uu = uu + vv * update_list[ii + 1]
        elif update_name == "edge":
            for ii, vv in enumerate(self.e_residual):
                uu = uu + vv * update_list[ii + 1]
        elif update_name == "angle":
            for ii, vv in enumerate(self.a_residual):
                uu = uu + vv * update_list[ii + 1]
        else:
            raise NotImplementedError
        return uu

    def list_update(self, update_list: list[Array], update_name: str = "node") -> Array:
        if self.update_style == "res_avg":
            return self.list_update_res_avg(update_list)
        elif self.update_style == "res_incr":
            return self.list_update_res_incr(update_list)
        elif self.update_style == "res_residual":
            return self.list_update_res_residual(update_list, update_name=update_name)
        else:
            raise RuntimeError(f"unknown update style {self.update_style}")

    def serialize(self) -> dict:
        data = {
            "@class": "HGNNLayer",
            "@version": 1,
            "e_rcut": self.e_rcut,
            "e_rcut_smth": self.e_rcut_smth,
            "e_sel": self.e_sel,
            "a_rcut": self.a_rcut,
            "a_rcut_smth": self.a_rcut_smth,
            "a_sel": self.a_sel,
            "ntypes": self.ntypes,
            "n_dim": self.n_dim,
            "e_dim": self.e_dim,
            "a_dim": self.a_dim,
            "axis_neuron": self.axis_neuron,
            "activation_function": self.activation_function,
            "update_style": self.update_style,
            "update_residual": self.update_residual,
            "update_residual_init": self.update_residual_init,
            "precision": self.precision,
            "use_hfea_v2e": self.use_hfea_v2e,
            "use_cross_order_v2e": self.use_cross_order_v2e,
            "phi_2": self.phi_2.serialize(),
            "rho_2": self.rho_2.serialize(),
            "phi_3": self.phi_3.serialize(),
            "rho_3": self.rho_3.serialize(),
            "psi_2": self.psi_2.serialize(),
            "psi_3": self.psi_3.serialize(),
            "psi_self": self.psi_self.serialize(),
            "psi_sym": self.psi_sym.serialize(),
        }
        if self.update_style == "res_residual":
            data.update(
                {
                    "@variables": {
                        "n_residual": [to_numpy_array(t) for t in self.n_residual],
                        "e_residual": [to_numpy_array(t) for t in self.e_residual],
                        "a_residual": [to_numpy_array(t) for t in self.a_residual],
                    }
                }
            )
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "HGNNLayer":
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        phi_2 = data.pop("phi_2")
        rho_2 = data.pop("rho_2")
        phi_3 = data.pop("phi_3")
        rho_3 = data.pop("rho_3")
        psi_2 = data.pop("psi_2")
        psi_3 = data.pop("psi_3")
        psi_self = data.pop("psi_self")
        psi_sym = data.pop("psi_sym")
        update_style = data["update_style"]
        variables = data.pop("@variables", {})
        n_residual = variables.get("n_residual", [])
        e_residual = variables.get("e_residual", [])
        a_residual = variables.get("a_residual", [])

        obj = cls(**data)
        obj.phi_2 = NativeLayer.deserialize(phi_2)
        obj.rho_2 = NativeLayer.deserialize(rho_2)
        obj.phi_3 = NativeLayer.deserialize(phi_3)
        obj.rho_3 = NativeLayer.deserialize(rho_3)
        obj.psi_2 = NativeLayer.deserialize(psi_2)
        obj.psi_3 = NativeLayer.deserialize(psi_3)
        obj.psi_self = NativeLayer.deserialize(psi_self)
        obj.psi_sym = NativeLayer.deserialize(psi_sym)

        if update_style == "res_residual":
            obj.n_residual = n_residual
            obj.e_residual = e_residual
            obj.a_residual = a_residual
        return obj


class DescrptBlockHGNN(NativeOP, DescriptorBlock):
    """HGNN descriptor block with message passing.

    Maintains node, edge (order-2), and angle (order-3) representations,
    and iteratively updates them through HGNNLayer instances.

    Parameters
    ----------
    e_rcut : float
        Edge cutoff radius.
    e_rcut_smth : float
        Edge smooth cutoff radius.
    e_sel : int
        Edge neighbor selection.
    a_rcut : float
        Angle cutoff radius.
    a_rcut_smth : float
        Angle smooth cutoff radius.
    a_sel : int
        Angle neighbor selection.
    ntypes : int
        Number of atom types.
    nlayers : int
        Number of HGNN layers.
    n_dim : int
        Node feature dimension.
    e_dim : int
        Edge feature dimension.
    a_dim : int
        Angle feature dimension.
    axis_neuron : int
        Axis neurons for symmetrization.
    activation_function : str
        Activation function.
    update_style : str
        Residual update style.
    update_residual : float
        Residual weight scale.
    update_residual_init : str
        Residual weight init mode.
    set_davg_zero : bool
        Set average to zero.
    exclude_types : list
        Excluded type pairs.
    env_protection : float
        Environment protection.
    precision : str
        Precision.
    use_exp_switch : bool
        Use exponential switch function.
    fix_stat_std : float
        Fixed standard deviation for normalization.
    seed : int or None
        Random seed.
    trainable : bool
        Whether trainable.
    """

    def __init__(
        self,
        e_rcut: float,
        e_rcut_smth: float,
        e_sel: int,
        a_rcut: float,
        a_rcut_smth: float,
        a_sel: int,
        ntypes: int,
        nlayers: int = 6,
        n_dim: int = 128,
        e_dim: int = 64,
        a_dim: int = 64,
        axis_neuron: int = 4,
        activation_function: str = "silu",
        update_style: str = "res_residual",
        update_residual: float = 0.1,
        update_residual_init: str = "const",
        set_davg_zero: bool = True,
        exclude_types: list[tuple[int, int]] = [],
        env_protection: float = 0.0,
        precision: str = "float64",
        use_exp_switch: bool = False,
        fix_stat_std: float = 0.3,
        seed: int | list[int] | None = None,
        trainable: bool = True,
        use_hfea_v2e: bool = False,
        use_cross_order_v2e: bool = False,
    ) -> None:
        super().__init__()
        self.e_rcut = float(e_rcut)
        self.e_rcut_smth = float(e_rcut_smth)
        self.e_sel = e_sel
        self.a_rcut = float(a_rcut)
        self.a_rcut_smth = float(a_rcut_smth)
        self.a_sel = a_sel
        self.ntypes = ntypes
        self.nlayers = nlayers
        sel = [e_sel] if isinstance(e_sel, int) else e_sel
        self.nnei = sum(sel)
        self.ndescrpt = self.nnei * 4
        assert len(sel) == 1
        self.sel = sel
        self.rcut = e_rcut
        self.rcut_smth = e_rcut_smth
        self.sec = self.sel
        self.split_sel = self.sel
        self.axis_neuron = axis_neuron
        self.set_davg_zero = set_davg_zero
        self.fix_stat_std = fix_stat_std
        self.set_stddev_constant = fix_stat_std != 0.0
        self.use_exp_switch = use_exp_switch
        self.use_hfea_v2e = use_hfea_v2e
        self.use_cross_order_v2e = use_cross_order_v2e

        self.n_dim = n_dim
        self.e_dim = e_dim
        self.a_dim = a_dim

        self.activation_function = activation_function
        self.update_style = update_style
        self.update_residual = update_residual
        self.update_residual_init = update_residual_init
        self.act = get_activation_fn(self.activation_function)
        self.prec = PRECISION_DICT[precision]

        self.reinit_exclude(exclude_types)
        self.env_protection = env_protection
        self.precision = precision
        self.epsilon = 1e-4
        self.seed = seed

        self.edge_embd = NativeLayer(
            1,
            self.e_dim,
            precision=precision,
            seed=child_seed(seed, 0),
            trainable=trainable,
        )
        self.angle_embd = NativeLayer(
            1,
            self.a_dim,
            precision=precision,
            bias=False,
            seed=child_seed(seed, 1),
            trainable=trainable,
        )
        layers = []
        for ii in range(nlayers):
            layers.append(
                HGNNLayer(
                    e_rcut=self.e_rcut,
                    e_rcut_smth=self.e_rcut_smth,
                    e_sel=self.sel,
                    a_rcut=self.a_rcut,
                    a_rcut_smth=self.a_rcut_smth,
                    a_sel=self.a_sel,
                    ntypes=self.ntypes,
                    n_dim=self.n_dim,
                    e_dim=self.e_dim,
                    a_dim=self.a_dim,
                    axis_neuron=self.axis_neuron,
                    activation_function=self.activation_function,
                    update_style=self.update_style,
                    update_residual=self.update_residual,
                    update_residual_init=self.update_residual_init,
                    precision=precision,
                    seed=child_seed(child_seed(seed, 2), ii),
                    trainable=trainable,
                    use_hfea_v2e=use_hfea_v2e,
                    use_cross_order_v2e=use_cross_order_v2e,
                )
            )
        self.layers = layers

        wanted_shape = (self.ntypes, self.nnei, 4)
        self.env_mat_edge = EnvMat(
            self.e_rcut,
            self.e_rcut_smth,
            protection=self.env_protection,
            use_exp_switch=self.use_exp_switch,
        )
        self.env_mat_angle = EnvMat(
            self.a_rcut,
            self.a_rcut_smth,
            protection=self.env_protection,
            use_exp_switch=self.use_exp_switch,
        )
        self.mean = np.zeros(wanted_shape, dtype=PRECISION_DICT[self.precision])
        self.stddev = np.ones(wanted_shape, dtype=PRECISION_DICT[self.precision])
        if self.set_stddev_constant:
            self.stddev = self.stddev * self.fix_stat_std

    def get_rcut(self) -> float:
        return self.e_rcut

    def get_rcut_smth(self) -> float:
        return self.e_rcut_smth

    def get_nsel(self) -> int:
        return sum(self.sel)

    def get_sel(self) -> list[int]:
        return self.sel

    def get_ntypes(self) -> int:
        return self.ntypes

    def get_dim_out(self) -> int:
        return self.dim_out

    def get_dim_in(self) -> int:
        return self.dim_in

    def get_dim_emb(self) -> int:
        return self.e_dim

    def __setitem__(self, key: str, value: Array) -> None:
        if key in ("avg", "data_avg", "davg"):
            self.mean = value
        elif key in ("std", "data_std", "dstd"):
            self.stddev = value
        else:
            raise KeyError(key)

    def __getitem__(self, key: str) -> Array:
        if key in ("avg", "data_avg", "davg"):
            return self.mean
        elif key in ("std", "data_std", "dstd"):
            return self.stddev
        else:
            raise KeyError(key)

    def mixed_types(self) -> bool:
        return True

    def get_env_protection(self) -> float:
        return self.env_protection

    @property
    def dim_out(self) -> int:
        return self.n_dim

    @property
    def dim_in(self) -> int:
        return self.n_dim

    @property
    def dim_emb(self) -> int:
        return self.get_dim_emb()

    def compute_input_stats(
        self,
        merged: Callable[[], list[dict]] | list[dict],
        path: DPPath | None = None,
    ) -> None:
        if self.set_stddev_constant and self.set_davg_zero:
            return
        env_mat_stat = EnvMatStatSe(self)
        if path is not None:
            path = path / env_mat_stat.get_hash()
        if path is None or not path.is_dir():
            if callable(merged):
                sampled = merged()
            else:
                sampled = merged
        else:
            sampled = []
        env_mat_stat.load_or_compute_stats(sampled, path)
        self.stats = env_mat_stat.stats
        mean, stddev = env_mat_stat()
        xp = array_api_compat.array_namespace(self.stddev)
        device = array_api_compat.device(self.stddev)
        if not self.set_davg_zero:
            self.mean = xp.asarray(
                mean, dtype=self.mean.dtype, copy=True, device=device
            )
        if not self.set_stddev_constant:
            self.stddev = xp.asarray(
                stddev, dtype=self.stddev.dtype, copy=True, device=device
            )

    def get_stats(self) -> dict[str, StatItem]:
        if self.stats is None:
            raise RuntimeError(
                "The statistics of the descriptor has not been computed."
            )
        return self.stats

    def reinit_exclude(
        self,
        exclude_types: list[tuple[int, int]] = [],
    ) -> None:
        self.exclude_types = exclude_types
        self.emask = PairExcludeMask(self.ntypes, exclude_types=exclude_types)

    def has_message_passing(self) -> bool:
        return True

    def need_sorted_nlist_for_lower(self) -> bool:
        return True

    def call(
        self,
        nlist: Array,
        coord_ext: Array,
        atype_ext: Array,
        atype_embd_ext: Array | None = None,
        mapping: Array | None = None,
        type_embedding: Array | None = None,
    ) -> tuple[Array, Array, Array, Array, Array]:
        xp = array_api_compat.array_namespace(nlist, coord_ext, atype_ext)
        nframes, nloc, nnei = nlist.shape
        nall = xp.reshape(coord_ext, (nframes, -1)).shape[1] // 3
        exclude_mask = self.emask.build_type_exclude_mask(nlist, atype_ext)
        exclude_mask = xp.astype(exclude_mask, xp.bool)
        nlist = xp.where(exclude_mask, nlist, xp.full_like(nlist, -1))

        # Edge environment matrix
        dmatrix, diff, sw = self.env_mat_edge.call(
            coord_ext,
            atype_ext,
            nlist,
            self.mean[...],
            self.stddev[...],
        )
        nlist_mask = nlist != -1
        sw = xp.reshape(sw, (nframes, nloc, nnei))
        sw = xp.where(nlist_mask, sw, xp.zeros_like(sw))

        # Angle neighbor list (subset within a_rcut)
        a_dist_mask = (safe_for_vector_norm(diff, axis=-1) < self.a_rcut)[
            :, :, : self.a_sel
        ]
        a_nlist = nlist[:, :, : self.a_sel]
        a_nlist = xp.where(a_dist_mask, a_nlist, xp.full_like(a_nlist, -1))

        _, a_diff, a_sw = self.env_mat_angle.call(
            coord_ext,
            atype_ext,
            a_nlist,
            self.mean[:, : self.a_sel, :],
            self.stddev[:, : self.a_sel, :],
        )

        a_nlist_mask = a_nlist != -1
        a_sw = xp.reshape(a_sw, (nframes, nloc, self.a_sel))
        a_sw = xp.where(a_nlist_mask, a_sw, xp.zeros_like(a_sw))

        # Set padding to 0 index
        nlist = xp.where(nlist == -1, xp.zeros_like(nlist), nlist)
        a_nlist = xp.where(a_nlist == -1, xp.zeros_like(a_nlist), a_nlist)

        # Node embedding
        atype_embd = atype_embd_ext[:, :nloc, :]
        assert list(atype_embd.shape) == [nframes, nloc, self.n_dim]
        node_ebd = self.act(atype_embd)

        # Edge input: 1/r and h2
        edge_input = dmatrix[:, :, :, :1]
        h2 = dmatrix[:, :, :, 1:]

        # Angle cosine matrix
        normalized_diff_i = a_diff / (
            safe_for_vector_norm(a_diff, axis=-1, keepdims=True) + 1e-6
        )
        normalized_diff_j = xp.matrix_transpose(normalized_diff_i)
        cosine_ij = xp.matmul(normalized_diff_i, normalized_diff_j) * (1 - 1e-6)

        # Angle embedding input
        angle_input = xp.reshape(
            cosine_ij, (nframes, nloc, self.a_sel, self.a_sel, 1)
        ) / (xp.pi**0.5)

        # Apply local mapping
        assert mapping is not None
        flat_map = xp.reshape(mapping, (nframes, -1))
        nlist = xp.reshape(
            xp_take_along_axis(flat_map, xp.reshape(nlist, (nframes, -1)), axis=1),
            nlist.shape,
        )

        # Edge and angle embeddings
        edge_ebd = self.act(self.edge_embd(edge_input))
        angle_ebd = self.angle_embd(angle_input)

        # Angle edge input (for phi_3)
        a_edge_input = edge_input[:, :, : self.a_sel, :]

        # Iterate through layers
        mapping_tiled = xp.tile(
            xp.reshape(mapping, (nframes, -1, 1)), (1, 1, self.n_dim)
        )
        for idx, ll in enumerate(self.layers):
            node_ebd_ext = xp_take_along_axis(node_ebd, mapping_tiled, axis=1)
            node_ebd, edge_ebd, angle_ebd = ll.call(
                node_ebd_ext,
                edge_ebd,
                h2,
                angle_ebd,
                nlist,
                nlist_mask,
                sw,
                a_nlist_mask,
                a_sw,
                edge_input,
                a_edge_input,
                cosine_ij,
            )

        # Final rotation matrix from h2g2
        h2g2 = _cal_hg(edge_ebd, h2, nlist_mask, sw)
        rot_mat = xp.matrix_transpose(h2g2)

        return (
            node_ebd,
            edge_ebd,
            h2,
            xp.reshape(rot_mat, (nframes, nloc, self.dim_emb, 3)),
            sw,
        )

    def serialize(self) -> dict:
        return {
            "e_rcut": self.e_rcut,
            "e_rcut_smth": self.e_rcut_smth,
            "e_sel": self.e_sel,
            "a_rcut": self.a_rcut,
            "a_rcut_smth": self.a_rcut_smth,
            "a_sel": self.a_sel,
            "ntypes": self.ntypes,
            "nlayers": self.nlayers,
            "n_dim": self.n_dim,
            "e_dim": self.e_dim,
            "a_dim": self.a_dim,
            "axis_neuron": self.axis_neuron,
            "activation_function": self.activation_function,
            "update_style": self.update_style,
            "update_residual": self.update_residual,
            "update_residual_init": self.update_residual_init,
            "set_davg_zero": self.set_davg_zero,
            "exclude_types": self.exclude_types,
            "env_protection": self.env_protection,
            "precision": self.precision,
            "use_exp_switch": self.use_exp_switch,
            "fix_stat_std": self.fix_stat_std,
            "use_hfea_v2e": self.use_hfea_v2e,
            "use_cross_order_v2e": self.use_cross_order_v2e,
            "edge_embd": self.edge_embd.serialize(),
            "angle_embd": self.angle_embd.serialize(),
            "hgnn_layers": [layer.serialize() for layer in self.layers],
            "env_mat": EnvMat(self.rcut, self.rcut_smth).serialize(),
            "@variables": {
                "davg": to_numpy_array(self.mean),
                "dstd": to_numpy_array(self.stddev),
            },
        }

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptBlockHGNN":
        data = data.copy()
        edge_embd = NativeLayer.deserialize(data.pop("edge_embd"))
        angle_embd = NativeLayer.deserialize(data.pop("angle_embd"))
        variables = data.pop("@variables")
        hgnn_layers = data.pop("hgnn_layers")
        data.pop("env_mat")
        obj = cls(**data)
        obj.edge_embd = edge_embd
        obj.angle_embd = angle_embd
        obj.mean = variables["davg"]
        obj.stddev = variables["dstd"]
        obj.layers = [HGNNLayer.deserialize(layer) for layer in hgnn_layers]
        return obj
