# SPDX-License-Identifier: LGPL-3.0-or-later

from deepmd.dpmodel.descriptor.hgnn import DescrptHGNN as DescrptHGNNDP
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.pt_expt.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.pt_expt.utils.update_sel import (
    UpdateSel,
)


@BaseDescriptor.register("hgnn")
@torch_module
class DescrptHGNN(DescrptHGNNDP):
    _update_sel_cls = UpdateSel
