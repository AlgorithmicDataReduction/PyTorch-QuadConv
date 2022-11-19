'''
'''

from torch_quadconv import MeshDataModule

'''
Extension of MeshDataModule.
'''

class DataModule(MeshDataModule):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        return

    # '''
    # Agglomerate feature batches.
    #
    # Input:
    #     data: all batched data
    # '''
    # def agglomerate(self, features):
    #     features = super().agglomerate(features)
    #
    #     print("Here")
    #
    #     if self.points == None:
    #         sq_shape = int(np.sqrt(features.shape[1]))
    #         features = features.reshape(features.shape[0], sq_shape, sq_shape, features.shape[-1])
    #
    #     return features
