'''
'''

from torch_quadconv import MeshDataModule

'''
Extension of MeshDataModule with a few extra bits.
'''

class DataModule(MeshDataModule):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
