# tackle different backend
from .depend import DependentBackend


def is_dgl():
    return DependentBackend.is_dgl()


class BackendOperator:
    """Operators based on different backends"""

    @staticmethod
    def mask(data, mask):
        return data.ndata[f"{mask}_mask"] if is_dgl() else data[f"{mask}_mask"]

    @staticmethod
    def label(data):
        return data.ndata["label"] if is_dgl() else data.y

    @staticmethod
    def feat(data):
        return data.ndata["feat"] if is_dgl() else data.x

    @staticmethod
    def gconv(op, data, feat):
        return op(data, feat) if is_dgl() else op(feat, data.edge_index)


bk_mask = BackendOperator.mask
bk_label = BackendOperator.label
bk_feat = BackendOperator.feat
bk_gconv = BackendOperator.gconv
