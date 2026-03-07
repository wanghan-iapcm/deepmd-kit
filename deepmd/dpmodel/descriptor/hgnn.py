# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import array_api_compat

from deepmd.dpmodel import (
    NativeOP,
)
from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.common import (
    cast_precision,
    to_numpy_array,
)
from deepmd.dpmodel.utils import (
    EnvMat,
)
from deepmd.dpmodel.utils.network import (
    NativeLayer,
)
from deepmd.dpmodel.utils.seed import (
    child_seed,
)
from deepmd.dpmodel.utils.type_embed import (
    TypeEmbedNet,
)
from deepmd.dpmodel.utils.update_sel import (
    UpdateSel,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.utils.finetune import (
    get_index_between_two_maps,
    map_pair_exclude_types,
)
from deepmd.utils.path import (
    DPPath,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .base_descriptor import (
    BaseDescriptor,
)
from .descriptor import (
    extend_descrpt_stat,
)
from .hgnn_block import (
    DescrptBlockHGNN,
    HGNNLayer,
)


class HGNNArgs:
    r"""Arguments for the HGNN descriptor block.

    Parameters
    ----------
    n_dim : int
        Node feature dimension.
    e_dim : int
        Order-2 (edge) hyperedge feature dimension.
    a_dim : int
        Order-3 (angle) hyperedge feature dimension.
    nlayers : int
        Number of HGNN layers.
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
    axis_neuron : int
        Axis neurons for symmetrization.
    update_style : str
        Residual update style.
    update_residual : float
        Residual weight scale.
    update_residual_init : str
        Residual weight init mode.
    fix_stat_std : float
        Fixed standard deviation.
    use_exp_switch : bool
        Use exponential switch function.
    use_hfea_v2e : bool
        Include existing hyperedge features in V→E rho functions.
    use_cross_order_v2e : bool
        Include edge features in the angle V→E phi function (cross-order).
    """

    def __init__(
        self,
        n_dim: int = 128,
        e_dim: int = 64,
        a_dim: int = 64,
        nlayers: int = 6,
        e_rcut: float = 6.0,
        e_rcut_smth: float = 5.0,
        e_sel: int = 120,
        a_rcut: float = 4.0,
        a_rcut_smth: float = 3.5,
        a_sel: int = 48,
        axis_neuron: int = 4,
        update_style: str = "res_residual",
        update_residual: float = 0.1,
        update_residual_init: str = "const",
        fix_stat_std: float = 0.3,
        use_exp_switch: bool = False,
        use_hfea_v2e: bool = False,
        use_cross_order_v2e: bool = False,
    ) -> None:
        self.n_dim = n_dim
        self.e_dim = e_dim
        self.a_dim = a_dim
        self.nlayers = nlayers
        self.e_rcut = e_rcut
        self.e_rcut_smth = e_rcut_smth
        self.e_sel = e_sel
        self.a_rcut = a_rcut
        self.a_rcut_smth = a_rcut_smth
        self.a_sel = a_sel
        self.axis_neuron = axis_neuron
        self.update_style = update_style
        self.update_residual = update_residual
        self.update_residual_init = update_residual_init
        self.fix_stat_std = fix_stat_std
        self.use_exp_switch = use_exp_switch
        self.use_hfea_v2e = use_hfea_v2e
        self.use_cross_order_v2e = use_cross_order_v2e

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(key)

    def serialize(self) -> dict:
        return {
            "n_dim": self.n_dim,
            "e_dim": self.e_dim,
            "a_dim": self.a_dim,
            "nlayers": self.nlayers,
            "e_rcut": self.e_rcut,
            "e_rcut_smth": self.e_rcut_smth,
            "e_sel": self.e_sel,
            "a_rcut": self.a_rcut,
            "a_rcut_smth": self.a_rcut_smth,
            "a_sel": self.a_sel,
            "axis_neuron": self.axis_neuron,
            "update_style": self.update_style,
            "update_residual": self.update_residual,
            "update_residual_init": self.update_residual_init,
            "fix_stat_std": self.fix_stat_std,
            "use_exp_switch": self.use_exp_switch,
            "use_hfea_v2e": self.use_hfea_v2e,
            "use_cross_order_v2e": self.use_cross_order_v2e,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "HGNNArgs":
        return cls(**data)


@BaseDescriptor.register("hgnn")
class DescrptHGNN(NativeOP, BaseDescriptor):
    r"""The HGNN descriptor.

    Implements a Hypergraph Neural Network descriptor that generalizes DPA3's
    line-graph architecture with configurable ordered hyperedges.

    Parameters
    ----------
    ntypes : int
        Number of atom types.
    hgnn : HGNNArgs or dict
        HGNN block arguments.
    concat_output_tebd : bool
        Whether to concatenate type embedding to output.
    activation_function : str
        Activation function.
    precision : str
        Precision.
    exclude_types : list
        Excluded type pairs.
    env_protection : float
        Environment protection.
    trainable : bool
        Whether trainable.
    seed : int or None
        Random seed.
    use_econf_tebd : bool
        Use electronic configuration type embedding.
    use_tebd_bias : bool
        Use bias in type embedding.
    type_map : list[str] or None
        Type map.
    """

    _update_sel_cls = UpdateSel

    def __init__(
        self,
        ntypes: int,
        hgnn: HGNNArgs | dict,
        concat_output_tebd: bool = False,
        activation_function: str = "silu",
        precision: str = "float64",
        exclude_types: list[tuple[int, int]] = [],
        env_protection: float = 0.0,
        trainable: bool = True,
        seed: int | list[int] | None = None,
        use_econf_tebd: bool = False,
        use_tebd_bias: bool = False,
        type_map: list[str] | None = None,
    ) -> None:
        super().__init__()

        def init_subclass_params(sub_data: dict | Any, sub_class: type) -> Any:
            if isinstance(sub_data, dict):
                return sub_class(**sub_data)
            elif isinstance(sub_data, sub_class):
                return sub_data
            else:
                raise ValueError(
                    f"Input args must be a {sub_class.__name__} class or a dict!"
                )

        self.hgnn_args = init_subclass_params(hgnn, HGNNArgs)
        self.activation_function = activation_function

        self.hgnn_block = DescrptBlockHGNN(
            self.hgnn_args.e_rcut,
            self.hgnn_args.e_rcut_smth,
            self.hgnn_args.e_sel,
            self.hgnn_args.a_rcut,
            self.hgnn_args.a_rcut_smth,
            self.hgnn_args.a_sel,
            ntypes,
            nlayers=self.hgnn_args.nlayers,
            n_dim=self.hgnn_args.n_dim,
            e_dim=self.hgnn_args.e_dim,
            a_dim=self.hgnn_args.a_dim,
            axis_neuron=self.hgnn_args.axis_neuron,
            activation_function=self.activation_function,
            update_style=self.hgnn_args.update_style,
            update_residual=self.hgnn_args.update_residual,
            update_residual_init=self.hgnn_args.update_residual_init,
            fix_stat_std=self.hgnn_args.fix_stat_std,
            use_exp_switch=self.hgnn_args.use_exp_switch,
            exclude_types=exclude_types,
            env_protection=env_protection,
            precision=precision,
            seed=child_seed(seed, 1),
            trainable=trainable,
            use_hfea_v2e=self.hgnn_args.use_hfea_v2e,
            use_cross_order_v2e=self.hgnn_args.use_cross_order_v2e,
        )

        self.use_econf_tebd = use_econf_tebd
        self.use_tebd_bias = use_tebd_bias
        self.type_map = type_map
        self.tebd_dim = self.hgnn_args.n_dim
        self.type_embedding = TypeEmbedNet(
            ntypes=ntypes,
            neuron=[self.tebd_dim],
            padding=True,
            activation_function="Linear",
            precision=precision,
            use_econf_tebd=self.use_econf_tebd,
            use_tebd_bias=use_tebd_bias,
            type_map=type_map,
            seed=child_seed(seed, 2),
            trainable=trainable,
        )
        self.concat_output_tebd = concat_output_tebd
        self.precision = precision
        self.exclude_types = exclude_types
        self.env_protection = env_protection
        self.trainable = trainable

        assert self.hgnn_block.e_rcut >= self.hgnn_block.a_rcut
        assert self.hgnn_block.e_sel >= self.hgnn_block.a_sel

        self.rcut = self.hgnn_block.get_rcut()
        self.rcut_smth = self.hgnn_block.get_rcut_smth()
        self.sel = self.hgnn_block.get_sel()
        self.ntypes = ntypes

    def get_rcut(self) -> float:
        return self.rcut

    def get_rcut_smth(self) -> float:
        return self.rcut_smth

    def get_nsel(self) -> int:
        return sum(self.sel)

    def get_sel(self) -> list[int]:
        return self.sel

    def get_ntypes(self) -> int:
        return self.ntypes

    def get_type_map(self) -> list[str]:
        return self.type_map

    def get_dim_out(self) -> int:
        ret = self.hgnn_block.dim_out
        if self.concat_output_tebd:
            ret += self.tebd_dim
        return ret

    def get_dim_emb(self) -> int:
        return self.hgnn_block.dim_emb

    def mixed_types(self) -> bool:
        return True

    def has_message_passing(self) -> bool:
        return self.hgnn_block.has_message_passing()

    def need_sorted_nlist_for_lower(self) -> bool:
        return True

    def get_env_protection(self) -> float:
        return self.hgnn_block.get_env_protection()

    def share_params(
        self, base_class: Any, shared_level: int, resume: bool = False
    ) -> None:
        raise NotImplementedError

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat: Any = None
    ) -> None:
        assert self.type_map is not None, (
            "'type_map' must be defined when performing type changing!"
        )
        remap_index, has_new_type = get_index_between_two_maps(self.type_map, type_map)
        self.type_map = type_map
        self.type_embedding.change_type_map(type_map=type_map)
        self.exclude_types = map_pair_exclude_types(self.exclude_types, remap_index)
        self.ntypes = len(type_map)
        block = self.hgnn_block
        if has_new_type:
            extend_descrpt_stat(
                block,
                type_map,
                des_with_stat=model_with_new_type_stat.hgnn_block
                if model_with_new_type_stat is not None
                else None,
            )
        block.ntypes = self.ntypes
        block.reinit_exclude(self.exclude_types)
        block["davg"] = block["davg"][remap_index]
        block["dstd"] = block["dstd"][remap_index]

    @property
    def dim_out(self) -> int:
        return self.get_dim_out()

    @property
    def dim_emb(self) -> int:
        return self.get_dim_emb()

    def compute_input_stats(
        self, merged: list[dict], path: DPPath | None = None
    ) -> None:
        descrpt_list = [self.hgnn_block]
        for ii, descrpt in enumerate(descrpt_list):
            descrpt.compute_input_stats(merged, path)

    def set_stat_mean_and_stddev(
        self,
        mean: list[Array],
        stddev: list[Array],
    ) -> None:
        descrpt_list = [self.hgnn_block]
        for ii, descrpt in enumerate(descrpt_list):
            descrpt.mean = mean[ii]
            descrpt.stddev = stddev[ii]

    def get_stat_mean_and_stddev(self) -> tuple[list[Array], list[Array]]:
        mean_list = [self.hgnn_block.mean]
        stddev_list = [self.hgnn_block.stddev]
        return mean_list, stddev_list

    @cast_precision
    def call(
        self,
        coord_ext: Array,
        atype_ext: Array,
        nlist: Array,
        mapping: Array | None = None,
    ) -> tuple[Array, Array, Array, Array, Array]:
        """Compute the descriptor.

        Parameters
        ----------
        coord_ext
            Extended coordinates. shape: nf x (nall x 3)
        atype_ext
            Extended atom types. shape: nf x nall
        nlist
            Neighbor list. shape: nf x nloc x nnei
        mapping
            Index mapping from extended to local region.

        Returns
        -------
        node_ebd
            Output descriptor. shape: nf x nloc x n_dim
        rot_mat
            Rotation matrix. shape: nf x nloc x e_dim x 3
        edge_ebd
            Edge embedding. shape: nf x nloc x nnei x e_dim
        h2
            Equivariant pair representation. shape: nf x nloc x nnei x 3
        sw
            Switch function. shape: nf x nloc x nnei
        """
        xp = array_api_compat.array_namespace(coord_ext, atype_ext, nlist)
        nframes, nloc, nnei = nlist.shape
        nall = xp.reshape(coord_ext, (nframes, -1)).shape[1] // 3

        type_embedding = self.type_embedding.call()
        node_ebd_ext = xp.reshape(
            xp.take(type_embedding, xp.reshape(atype_ext[:, :nloc], (-1,)), axis=0),
            (nframes, nloc, self.tebd_dim),
        )
        node_ebd_inp = node_ebd_ext[:, :nloc, :]

        node_ebd, edge_ebd, h2, rot_mat, sw = self.hgnn_block(
            nlist,
            coord_ext,
            atype_ext,
            node_ebd_ext,
            mapping,
        )
        if self.concat_output_tebd:
            node_ebd = xp.concat([node_ebd, node_ebd_inp], axis=-1)
        return node_ebd, rot_mat, edge_ebd, h2, sw

    def serialize(self) -> dict:
        block = self.hgnn_block
        data = {
            "@class": "Descriptor",
            "type": "hgnn",
            "@version": 1,
            "ntypes": self.ntypes,
            "hgnn_args": self.hgnn_args.serialize(),
            "concat_output_tebd": self.concat_output_tebd,
            "activation_function": self.activation_function,
            "precision": self.precision,
            "exclude_types": self.exclude_types,
            "env_protection": self.env_protection,
            "trainable": self.trainable,
            "use_econf_tebd": self.use_econf_tebd,
            "use_tebd_bias": self.use_tebd_bias,
            "type_map": self.type_map,
            "type_embedding": self.type_embedding.serialize(),
        }
        hgnn_variable = {
            "edge_embd": block.edge_embd.serialize(),
            "angle_embd": block.angle_embd.serialize(),
            "hgnn_layers": [layer.serialize() for layer in block.layers],
            "env_mat": EnvMat(block.rcut, block.rcut_smth).serialize(),
            "@variables": {
                "davg": to_numpy_array(block["davg"]),
                "dstd": to_numpy_array(block["dstd"]),
            },
        }
        data.update({"hgnn_variable": hgnn_variable})
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "DescrptHGNN":
        data = data.copy()
        version = data.pop("@version")
        check_version_compatibility(version, 1, 1)
        data.pop("@class")
        data.pop("type")
        hgnn_variable = data.pop("hgnn_variable").copy()
        type_embedding = data.pop("type_embedding")
        data["hgnn"] = HGNNArgs(**data.pop("hgnn_args"))
        obj = cls(**data)
        obj.type_embedding = TypeEmbedNet.deserialize(type_embedding)

        statistic = hgnn_variable.pop("@variables")
        env_mat = hgnn_variable.pop("env_mat")
        hgnn_layers = hgnn_variable.pop("hgnn_layers")
        obj.hgnn_block.edge_embd = NativeLayer.deserialize(
            hgnn_variable.pop("edge_embd")
        )
        obj.hgnn_block.angle_embd = NativeLayer.deserialize(
            hgnn_variable.pop("angle_embd")
        )
        obj.hgnn_block["davg"] = statistic["davg"]
        obj.hgnn_block["dstd"] = statistic["dstd"]
        obj.hgnn_block.layers = [HGNNLayer.deserialize(layer) for layer in hgnn_layers]
        return obj

    @classmethod
    def update_sel(
        cls,
        train_data: DeepmdDataSystem,
        type_map: list[str] | None,
        local_jdata: dict,
    ) -> tuple[Array, Array]:
        local_jdata_cpy = local_jdata.copy()
        update_sel = cls._update_sel_cls()
        min_nbor_dist, hgnn_e_sel = update_sel.update_one_sel(
            train_data,
            type_map,
            local_jdata_cpy["hgnn"]["e_rcut"],
            local_jdata_cpy["hgnn"]["e_sel"],
            True,
        )
        local_jdata_cpy["hgnn"]["e_sel"] = hgnn_e_sel[0]

        min_nbor_dist, hgnn_a_sel = update_sel.update_one_sel(
            train_data,
            type_map,
            local_jdata_cpy["hgnn"]["a_rcut"],
            local_jdata_cpy["hgnn"]["a_sel"],
            True,
        )
        local_jdata_cpy["hgnn"]["a_sel"] = hgnn_a_sel[0]

        return local_jdata_cpy, min_nbor_dist
