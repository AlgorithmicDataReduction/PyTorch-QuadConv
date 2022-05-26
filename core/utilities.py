'''
'''
import torch
from pytorch_lightning.callbacks.progress import TQDMProgressBar

'''
Sobolev loss function

Input:
    pred :
    x :
    order :
    lambda_r :
'''
def sobolev_loss(pred, x, order=1, lambda_r=(0.25, 0.0625)):
    #setup
    bs = pred.shape[0]

    sq_shape = np.sqrt(x.shape[2]).astype(int)
    numel = sq_shape * sq_shape

    temp_x = torch.reshape(x, (x.shape[0],x.shape[1],sq_shape,sq_shape))
    temp_pred = torch.reshape(pred, (pred.shape[0],pred.shape[1],sq_shape,sq_shape))

    #compute function l1 error
    loss = torch.sum((temp_pred-temp_x)**2)

    #compute derivatives l1 error
    stencil = torch.tensor([[0.0, -1.0, 0.0],[-1.0, 4.0, -1.0],[0.0, -1.0, 0.0]], device=x.device)*1/4
    stencil = torch.reshape(stencil, (1,1,3,3)).repeat(1, x.shape[1], 1, 1)

    for i in range(order):
        temp_x = torch.nn.functional.conv2d(temp_x, stencil)
        temp_pred = torch.nn.functional.conv2d(temp_pred, stencil)

        loss += lambda_r[i] * torch.sum((temp_pred-temp_x)**2)

    return loss/bs

'''
'''
def make_gif(model, data_module, save_path):
    train, test, s = data_module.get_data()
    size = data_module.size
    tile = data_module.num_tiles

    model.eval()
    with torch.no_grad():
        processed = model(train)

    processed_squares = processed.reshape(-1, size, size).reshape(-1, tile, tile, size, size)

    processed_full = torch.zeros(450, size*tile, size*tile)

    for i in range(450):
        for j in range(tile):
            for k in range(tile):
                processed_full[i,size*j:size*(j+1),size*k:size*(k+1)] = processed_squares[i,j,k,:,:]

    filenames = []
    fig1, ax1 = plt.subplots()

    with torch.no_grad():
        for i in range(450):
            ax1.imshow(processed_full[i,:,:], vmin=-1, vmax=1)
            filename = f'{i}.png'
            filenames.append(filename)
            plt.savefig(filename)
            plt.cla()

        plt.close('all')
        image_list = []
        for filename in filenames:
            image = imageio.imread(filename)
            image_list.append(image)

        imageio.mimwrite(os.path.join(save_path , 'processed_quadconv_new.gif'), image_list)

        for filename in set(filenames):
            os.remove(filename)

    return

'''
'''
class ProgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__()
