# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
    List,
    Optional,
)

import torch

from deepmd.dpmodel import (
    ModelOutputDef,
    get_hessian_name,
)
from deepmd.pt.model.model.transform_output import (
    communicate_extended_output,
    fit_output_to_model_output,
)
from deepmd.pt.utils.nlist import (
    build_neighbor_list,
    extend_coord_with_ghosts,
    nlist_distinguish_types,
)
from deepmd.pt.utils.region import (
    normalize_coord,
)


def make_model(T_AtomicModel):
    """Make a model as a derived class of an atomic model.

    The model provide two interfaces.

    1. the `forward_common_lower`, that takes extended coordinates, atyps and neighbor list,
    and outputs the atomic and property and derivatives (if required) on the extended region.

    2. the `forward_common`, that takes coordinates, atypes and cell and predicts
    the atomic and reduced property, and derivatives (if required) on the local region.

    Parameters
    ----------
    T_AtomicModel
        The atomic model.

    Returns
    -------
    CM
        The model.

    """

    class CM(T_AtomicModel):
        def __init__(
            self,
            *args,
            **kwargs,
        ):
            super().__init__(
                *args,
                **kwargs,
            )

        def model_output_def(self):
            """Get the output def for the model."""
            return ModelOutputDef(self.fitting_output_def())

        # Cannot use the name forward. torch script does not work
        #
        # Wrapper for computing hessian. We only provide hessian calculation
        # for the forward interface, not for forward_lower
        #
        # Low efficiency: hessian is computed from scratch, we do not use
        # the jacobian.
        @torch.jit.export
        def forward_common(
            self,
            coord,
            atype,
            box: Optional[torch.Tensor] = None,
            fparam: Optional[torch.Tensor] = None,
            aparam: Optional[torch.Tensor] = None,
            do_atomic_virial: bool = False,
        ) -> Dict[str, torch.Tensor]:
            """Return model prediction.

            Parameters
            ----------
            coord
                The coordinates of the atoms.
                shape: nf x (nloc x 3)
            atype
                The type of atoms. shape: nf x nloc
            box
                The simulation box. shape: nf x 9
            fparam
                The frame parameter. shape: nf x nfp
            aparam
                The atomic parameter. shape: nf x (nloc x nap)
            do_atomic_virial
                If calculate the atomic virial.

            Returns
            -------
            ret_dict
                The result dict of type Dict[str,torch.Tensor].
                The keys are defined by the `ModelOutputDef`.

            """
            ret = self.forward_common_(
                coord,
                atype,
                box=box,
                fparam=fparam,
                aparam=aparam,
                do_atomic_virial=do_atomic_virial,
            )
            vdef = self.fitting_output_def()
            hess_yes = [vdef[kk].r_hessian for kk in vdef.keys()]
            if any(hess_yes):
                hess = self._cal_hessian_all(
                    coord,
                    atype,
                    box=box,
                    fparam=fparam,
                    aparam=aparam,
                )
                ret.update(hess)
            return ret

        # cannot use the name forward. torch script does not work
        def forward_common_(
            self,
            coord,
            atype,
            box: Optional[torch.Tensor] = None,
            fparam: Optional[torch.Tensor] = None,
            aparam: Optional[torch.Tensor] = None,
            do_atomic_virial: bool = False,
        ) -> Dict[str, torch.Tensor]:
            """Return model prediction.

            Parameters
            ----------
            coord
                The coordinates of the atoms.
                shape: nf x (nloc x 3)
            atype
                The type of atoms. shape: nf x nloc
            box
                The simulation box. shape: nf x 9. if None no-pbc is assumed
            fparam
                The frame parameter. shape: nf x nfp
            aparam
                The atomic parameter. shape: nf x (nloc x nap)
            do_atomic_virial
                If calculate the atomic virial.

            Returns
            -------
            ret_dict
                The result dict of type Dict[str,torch.Tensor].
                The keys are defined by the `ModelOutputDef`.

            """
            nframes, nloc = atype.shape[:2]
            if box is not None:
                coord_normalized = normalize_coord(
                    coord.view(nframes, nloc, 3),
                    box.reshape(nframes, 3, 3),
                )
            else:
                coord_normalized = coord.clone()
            extended_coord, extended_atype, mapping = extend_coord_with_ghosts(
                coord_normalized, atype, box, self.get_rcut()
            )
            nlist = build_neighbor_list(
                extended_coord,
                extended_atype,
                nloc,
                self.get_rcut(),
                self.get_sel(),
                distinguish_types=self.distinguish_types(),
            )
            extended_coord = extended_coord.view(nframes, -1, 3)
            model_predict_lower = self.forward_common_lower(
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                do_atomic_virial=do_atomic_virial,
                fparam=fparam,
                aparam=aparam,
            )
            model_predict = communicate_extended_output(
                model_predict_lower,
                self.model_output_def(),
                mapping,
                do_atomic_virial=do_atomic_virial,
            )
            return model_predict

        @torch.jit.export
        def forward_common_lower(
            self,
            extended_coord,
            extended_atype,
            nlist,
            mapping: Optional[torch.Tensor] = None,
            fparam: Optional[torch.Tensor] = None,
            aparam: Optional[torch.Tensor] = None,
            do_atomic_virial: bool = False,
        ):
            """Return model prediction. Lower interface that takes
            extended atomic coordinates and types, nlist, and mapping
            as input, and returns the predictions on the extended region.
            The predictions are not reduced.

            Parameters
            ----------
            extended_coord
                coodinates in extended region. nf x (nall x 3)
            extended_atype
                atomic type in extended region. nf x nall
            nlist
                neighbor list. nf x nloc x nsel.
            mapping
                mapps the extended indices to local indices. nf x nall.
            fparam
                The frame parameter. shape: nf x nfp
            aparam
                The atomic parameter. shape: nf x (nloc x nap)
            do_atomic_virial
                whether calculate atomic virial.

            Returns
            -------
            result_dict
                the result dict, defined by the `FittingOutputDef`.

            """
            nframes, nall = extended_atype.shape[:2]
            extended_coord = extended_coord.view(nframes, -1, 3)
            nlist = self.format_nlist(extended_coord, extended_atype, nlist)
            atomic_ret = self.forward_atomic(
                extended_coord,
                extended_atype,
                nlist,
                mapping=mapping,
                fparam=fparam,
                aparam=aparam,
            )
            model_predict = fit_output_to_model_output(
                atomic_ret,
                self.fitting_output_def(),
                extended_coord,
                do_atomic_virial=do_atomic_virial,
            )
            return model_predict

        def format_nlist(
            self,
            extended_coord: torch.Tensor,
            extended_atype: torch.Tensor,
            nlist: torch.Tensor,
        ):
            """Format the neighbor list.

            1. If the number of neighbors in the `nlist` is equal to sum(self.sel),
            it does nothong

            2. If the number of neighbors in the `nlist` is smaller than sum(self.sel),
            the `nlist` is pad with -1.

            3. If the number of neighbors in the `nlist` is larger than sum(self.sel),
            the nearest sum(sel) neighbors will be preseved.

            Known limitations:

            In the case of self.distinguish_types, the nlist is always formatted.
            May have side effact on the efficiency.

            Parameters
            ----------
            extended_coord
                coodinates in extended region. nf x nall x 3
            extended_atype
                atomic type in extended region. nf x nall
            nlist
                neighbor list. nf x nloc x nsel

            Returns
            -------
            formated_nlist
                the formated nlist.

            """
            distinguish_types = self.distinguish_types()
            nlist = self._format_nlist(extended_coord, nlist, sum(self.get_sel()))
            if distinguish_types:
                nlist = nlist_distinguish_types(nlist, extended_atype, self.get_sel())
            return nlist

        def _format_nlist(
            self,
            extended_coord: torch.Tensor,
            nlist: torch.Tensor,
            nnei: int,
        ):
            n_nf, n_nloc, n_nnei = nlist.shape
            # nf x nall x 3
            extended_coord = extended_coord.view([n_nf, -1, 3])
            rcut = self.get_rcut()

            if n_nnei < nnei:
                nlist = torch.cat(
                    [
                        nlist,
                        -1
                        * torch.ones(
                            [n_nf, n_nloc, nnei - n_nnei], dtype=nlist.dtype
                        ).to(nlist.device),
                    ],
                    dim=-1,
                )
            elif n_nnei > nnei:
                m_real_nei = nlist >= 0
                nlist = torch.where(m_real_nei, nlist, 0)
                # nf x nloc x 3
                coord0 = extended_coord[:, :n_nloc, :]
                # nf x (nloc x nnei) x 3
                index = nlist.view(n_nf, n_nloc * n_nnei, 1).expand(-1, -1, 3)
                coord1 = torch.gather(extended_coord, 1, index)
                # nf x nloc x nnei x 3
                coord1 = coord1.view(n_nf, n_nloc, n_nnei, 3)
                # nf x nloc x nnei
                rr = torch.linalg.norm(coord0[:, :, None, :] - coord1, dim=-1)
                rr = torch.where(m_real_nei, rr, float("inf"))
                rr, nlist_mapping = torch.sort(rr, dim=-1)
                nlist = torch.gather(nlist, 2, nlist_mapping)
                nlist = torch.where(rr > rcut, -1, nlist)
                nlist = nlist[..., :nnei]
            else:  # n_nnei == nnei:
                pass  # great!
            assert nlist.shape[-1] == nnei
            return nlist

        ## FAILED TO JIT this method
        ## torch/autograd/functional does not support jit script.
        ## tested torch version: 2.2
        @torch.jit.ignore
        def _cal_hessian_all(
            self,
            coord: torch.Tensor,
            atype: torch.Tensor,
            box: Optional[torch.Tensor] = None,
            fparam: Optional[torch.Tensor] = None,
            aparam: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            nf, nloc = atype.shape
            coord = coord.view([nf, (nloc * 3)])
            box = box.view([nf, 9]) if box is not None else None
            fparam = fparam.view([nf, -1]) if fparam is not None else None
            aparam = aparam.view([nf, nloc, -1]) if aparam is not None else None
            fdef = self.fitting_output_def()
            # keys of values that require hessian
            hess_keys: List[str] = []
            for kk in fdef.keys():
                if fdef[kk].r_hessian:
                    hess_keys.append(kk)
            # result dict init by empty lists
            res = {get_hessian_name(kk): [] for kk in hess_keys}
            # loop over variable
            for kk in hess_keys:
                vdef = fdef[kk]
                vshape = vdef.shape
                vsize = 1
                for ii in vshape:
                    vsize *= ii
                # loop over frames
                for ii in range(nf):
                    icoord = coord[ii]
                    iatype = atype[ii]
                    ibox = box[ii] if box is not None else None
                    ifparam = fparam[ii] if fparam is not None else None
                    iaparam = aparam[ii] if aparam is not None else None
                    # loop over all components
                    for idx in range(vsize):
                        hess = self._cal_hessian_one_component(
                            idx, icoord, iatype, ibox, ifparam, iaparam
                        )
                        res[get_hessian_name(kk)].append(hess)
                res[get_hessian_name(kk)] = torch.stack(res[get_hessian_name(kk)]).view(
                    (nf, *vshape, nloc * 3, nloc * 3)
                )
            return res

        def _cal_hessian_one_component(
            self,
            ci,
            coord,
            atype,
            box: Optional[torch.Tensor] = None,
            fparam: Optional[torch.Tensor] = None,
            aparam: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            # coord, # (nloc x 3)
            # atype, # nloc
            # box: Optional[torch.Tensor] = None,     # 9
            # fparam: Optional[torch.Tensor] = None,  # nfp
            # aparam: Optional[torch.Tensor] = None,  # (nloc x nap)
            wc = wrapper_class_forward_energy(self, ci, atype, box, fparam, aparam)

            hess = torch.autograd.functional.hessian(
                wc,
                coord,
                create_graph=False,
            )
            return hess

    class wrapper_class_forward_energy:
        def __init__(
            self,
            obj: CM,
            ci: int,
            atype: torch.Tensor,
            box: Optional[torch.Tensor],
            fparam: Optional[torch.Tensor],
            aparam: Optional[torch.Tensor],
        ):
            self.atype, self.box, self.fparam, self.aparam = atype, box, fparam, aparam
            self.ci = ci
            self.obj = obj

        def __call__(
            self,
            xx,
        ):
            ci = self.ci
            atype, box, fparam, aparam = self.atype, self.box, self.fparam, self.aparam
            res = self.obj.forward_common_(
                xx.unsqueeze(0),
                atype.unsqueeze(0),
                box.unsqueeze(0) if box is not None else None,
                fparam.unsqueeze(0) if fparam is not None else None,
                aparam.unsqueeze(0) if aparam is not None else None,
                do_atomic_virial=False,
            )
            er = res["energy_redu"][0].view([-1])[ci]
            return er

    return CM
