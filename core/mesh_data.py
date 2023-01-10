'''
'''

from core.torch_quadconv import MeshDataModule

'''
Extension of MeshDataModule.
'''

class DataModule(MeshDataModule):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        return
