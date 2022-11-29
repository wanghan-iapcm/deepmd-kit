from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, List, Tuple

import numpy as np
from deepmd.env import tf
from deepmd.utils import Plugin, PluginVariant


class Descriptor(PluginVariant):
    r"""The abstract class for descriptors. All specific descriptors should
    be based on this class.

    The descriptor :math:`\mathcal{D}` describes the environment of an atom,
    which should be a function of coordinates and types of its neighbour atoms.

    Examples
    --------
    >>> descript = Descriptor(type="se_e2_a", rcut=6., rcut_smth=0.5, sel=[50])
    >>> type(descript)
    <class 'deepmd.descriptor.se_a.DescrptSeA'>

    Notes
    -----
    Only methods and attributes defined in this class are generally public,
    that can be called by other classes.
    """

    __plugins = Plugin()

    @staticmethod
    def register(key: str) -> "Descriptor":
        """Regiester a descriptor plugin.

        Parameters
        ----------
        key : str
            the key of a descriptor

        Returns
        -------
        Descriptor
            the regiestered descriptor

        Examples
        --------
        >>> @Descriptor.register("some_descrpt")
            class SomeDescript(Descriptor):
                pass
        """
        return Descriptor.__plugins.register(key)

    def __new__(cls, *args, **kwargs):
        if cls is Descriptor:
            try:
                descrpt_type = kwargs['type']
            except KeyError:
                raise KeyError('the type of descriptor should be set by `type`')
            if descrpt_type in Descriptor.__plugins.plugins:
                cls = Descriptor.__plugins.plugins[descrpt_type]
            else:
                raise RuntimeError('Unknown descriptor type: ' + descrpt_type)
        return super().__new__(cls)

    @abstractmethod
    def get_rcut(self) -> float:
        """
        Returns the cut-off radius.

        Returns
        -------
        float
            the cut-off radius

        Notes
        -----
        This method must be implemented, as it's called by other classes.
        """

    @abstractmethod
    def get_ntypes(self) -> int:
        """
        Returns the number of atom types.

        Returns
        -------
        int
            the number of atom types

        Notes
        -----
        This method must be implemented, as it's called by other classes.
        """

    @abstractmethod
    def get_dim_out(self) -> int:
        """
        Returns the output dimension of this descriptor.

        Returns
        -------
        int
            the output dimension of this descriptor

        Notes
        -----
        This method must be implemented, as it's called by other classes.
        """

    def get_dim_rot_mat_1(self) -> int:
        """
        Returns the first dimension of the rotation matrix. The rotation is of shape
        dim_1 x 3

        Returns
        -------
        int
            the first dimension of the rotation matrix
        """
        # TODO: I think this method should be implemented as it's called by dipole and
        # polar fitting network. However, currently not all descriptors have this
        # method.
        raise NotImplementedError

    def get_nlist(self) -> Tuple[tf.Tensor, tf.Tensor, List[int], List[int]]:
        """
        Returns neighbor information.

        Returns
        -------
        nlist : tf.Tensor
            Neighbor list
        rij : tf.Tensor
            The relative distance between the neighbor and the center atom.
        sel_a : list[int]
            The number of neighbors with full information
        sel_r : list[int]
            The number of neighbors with only radial information
        """
        # TODO: I think this method should be implemented as it's called by energy
        # model. However, se_ar and hybrid doesn't have this method.
        raise NotImplementedError

    @abstractmethod
    def compute_input_stats(self,
                            data_coord: List[np.ndarray],
                            data_box: List[np.ndarray],
                            data_atype: List[np.ndarray],
                            natoms_vec: List[np.ndarray],
                            mesh: List[np.ndarray],
                            input_dict: Dict[str, List[np.ndarray]]
                            ) -> None:
        """
        Compute the statisitcs (avg and std) of the training data. The input will be
        normalized by the statistics.

        Parameters
        ----------
        data_coord : list[np.ndarray]
            The coordinates. Can be generated by
            :meth:`deepmd.model.model_stat.make_stat_input`
        data_box : list[np.ndarray]
            The box. Can be generated by
            :meth:`deepmd.model.model_stat.make_stat_input`
        data_atype : list[np.ndarray]
            The atom types. Can be generated by :meth:`deepmd.model.model_stat.make_stat_input`
        natoms_vec : list[np.ndarray]
            The vector for the number of atoms of the system and different types of
            atoms. Can be generated by :meth:`deepmd.model.model_stat.make_stat_input`
        mesh : list[np.ndarray]
            The mesh for neighbor searching. Can be generated by
            :meth:`deepmd.model.model_stat.make_stat_input`
        input_dict : dict[str, list[np.ndarray]]
            Dictionary for additional input

        Notes
        -----
        This method must be implemented, as it's called by other classes.
        """

    @abstractmethod
    def build(self,
              coord_: tf.Tensor,
              atype_: tf.Tensor,
              natoms: tf.Tensor,
              box_: tf.Tensor,
              mesh: tf.Tensor,
              input_dict: Dict[str, Any],
              reuse: Optional[bool] = None,
              suffix: str = '',
              ) -> tf.Tensor:
        """
        Build the computational graph for the descriptor.

        Parameters
        ----------
        coord_ : tf.Tensor
            The coordinate of atoms
        atype_ : tf.Tensor
            The type of atoms
        natoms : tf.Tensor
            The number of atoms. This tensor has the length of Ntypes + 2
            natoms[0]: number of local atoms
            natoms[1]: total number of atoms held by this processor
            natoms[i]: 2 <= i < Ntypes+2, number of type i atoms
        box : tf.Tensor
            The box of frames
        mesh : tf.Tensor
            For historical reasons, only the length of the Tensor matters.
            if size of mesh == 6, pbc is assumed.
            if size of mesh == 0, no-pbc is assumed.
        input_dict : dict[str, Any]
            Dictionary for additional inputs
        reuse : bool, optional
            The weights in the networks should be reused when get the variable.
        suffix : str, optional
            Name suffix to identify this descriptor

        Returns
        -------
        descriptor: tf.Tensor
            The output descriptor

        Notes
        -----
        This method must be implemented, as it's called by other classes.
        """

    def enable_compression(self,
                           min_nbor_dist: float,
                           model_file: str = 'frozon_model.pb',
                           table_extrapolate: float = 5.,
                           table_stride_1: float = 0.01,
                           table_stride_2: float = 0.1,
                           check_frequency: int = -1,
                           suffix: str = "",
                           ) -> None:
        """
        Reveive the statisitcs (distance, max_nbor_size and env_mat_range) of the
        training data.

        Parameters
        ----------
        min_nbor_dist : float
                The nearest distance between atoms
        model_file : str, default: 'frozon_model.pb'
                The original frozen model, which will be compressed by the program
        table_extrapolate : float, default: 5.
                The scale of model extrapolation
        table_stride_1 : float, default: 0.01
                The uniform stride of the first table
        table_stride_2 : float, default: 0.1
                The uniform stride of the second table
        check_frequency : int, default: -1
                The overflow check frequency
        suffix : str, optional
                The suffix of the scope

        Notes
        -----
        This method is called by others when the descriptor supported compression.
        """
        raise NotImplementedError(
            "Descriptor %s doesn't support compression!" % type(self).__name__)

    def enable_mixed_precision(self, mixed_prec: Optional[dict] = None) -> None:
        """
        Reveive the mixed precision setting.

        Parameters
        ----------
        mixed_prec
                The mixed precision setting used in the embedding net

        Notes
        -----
        This method is called by others when the descriptor supported compression.
        """
        raise NotImplementedError(
            "Descriptor %s doesn't support mixed precision training!"
            % type(self).__name__
        )

    @abstractmethod
    def prod_force_virial(self,
                          atom_ener: tf.Tensor,
                          natoms: tf.Tensor
                          ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Compute force and virial.

        Parameters
        ----------
        atom_ener : tf.Tensor
            The atomic energy
        natoms : tf.Tensor
            The number of atoms. This tensor has the length of Ntypes + 2
            natoms[0]: number of local atoms
            natoms[1]: total number of atoms held by this processor
            natoms[i]: 2 <= i < Ntypes+2, number of type i atoms

        Returns
        -------
        force : tf.Tensor
            The force on atoms
        virial : tf.Tensor
            The total virial
        atom_virial : tf.Tensor
            The atomic virial
        """

    def get_feed_dict(self,
                      coord_: tf.Tensor,
                      atype_: tf.Tensor,
                      natoms: tf.Tensor,
                      box: tf.Tensor,
                      mesh: tf.Tensor
                      ) -> Dict[str, tf.Tensor]:
        """
        Generate the feed_dict for current descriptor

        Parameters
        ----------
        coord_ : tf.Tensor
            The coordinate of atoms
        atype_ : tf.Tensor
            The type of atoms
        natoms : tf.Tensor
            The number of atoms. This tensor has the length of Ntypes + 2
            natoms[0]: number of local atoms
            natoms[1]: total number of atoms held by this processor
            natoms[i]: 2 <= i < Ntypes+2, number of type i atoms
        box : tf.Tensor
            The box. Can be generated by deepmd.model.make_stat_input
        mesh : tf.Tensor
            For historical reasons, only the length of the Tensor matters.
            if size of mesh == 6, pbc is assumed. 
            if size of mesh == 0, no-pbc is assumed. 

        Returns
        -------
        feed_dict : dict[str, tf.Tensor]
            The output feed_dict of current descriptor
        """
        feed_dict = {
            't_coord:0'  :coord_,
            't_type:0'   :atype_,
            't_natoms:0' :natoms,
            't_box:0'    :box,
            't_mesh:0'   :mesh
        }
        return feed_dict

    def init_variables(self,
                       graph: tf.Graph,
                       graph_def: tf.GraphDef,
                       suffix : str = "",
    ) -> None:
        """
        Init the embedding net variables with the given dict

        Parameters
        ----------
        graph : tf.Graph
            The input frozen model graph
        graph_def : tf.GraphDef
            The input frozen model graph_def
        suffix : str, optional
            The suffix of the scope
        
        Notes
        -----
        This method is called by others when the descriptor supported initialization from the given variables.
        """
        raise NotImplementedError(
            "Descriptor %s doesn't support initialization from the given variables!" % type(self).__name__)

    def get_tensor_names(self, suffix : str = "") -> Tuple[str]:
        """Get names of tensors.
        
        Parameters
        ----------
        suffix : str
            The suffix of the scope

        Returns
        -------
        Tuple[str]
            Names of tensors
        """
        raise NotImplementedError("Descriptor %s doesn't support this property!" % type(self).__name__)

    def pass_tensors_from_frz_model(self,
                                    *tensors : tf.Tensor,
    ) -> None:
        """
        Pass the descrpt_reshape tensor as well as descrpt_deriv tensor from the frz graph_def

        Parameters
        ----------
        *tensors : tf.Tensor
            passed tensors
        
        Notes
        -----
        The number of parameters in the method must be equal to the numbers of returns in
        :meth:`get_tensor_names`.
        """
        raise NotImplementedError("Descriptor %s doesn't support this method!" % type(self).__name__)
