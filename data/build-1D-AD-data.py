from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm

import numpy as np
from numba import njit

@njit
def generate_time_series(time,x,c,f_0,f):

    tseries = np.zeros((len(time),len(x)))

    for i,t in enumerate(time):

        tseries[i,] = f(t,x,c,f_0)

    return tseries

@njit
def advection_diffusion_zero_visc(t , x , c , v = 0.05 , terms = 100):
    ''' 
    Closed form solution for u(x,0) = -sin(pi*x) and zero boundary condtions
    '''

    sol =  np.zeros_like(x)

    for p in range(terms):

        sol += (-1)**p * ((2*p*np.sin(np.pi*p*x) + (2*p+1)*np.cos(((2*p+1)/2) * np.pi * x)))

    mult_fac =  (v/c)**3 * np.exp((c/(2*v))*((x+1) - (c/2)*t))

    return mult_fac * sol



if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--tspan" , type=int , default=100 , help="temporal size")
    parser.add_argument("--xspan" , type=int , default=100 , help="spatial size")
    parser.add_argument("--aspan" , type=int , default=100 , help="initial value")
    parser.add_argument("--savepath", type=str, default='./data' , help='save path for the data')

    args, _ = parser.parse_known_args()

    x = np.linspace(start=-1, stop=1, num = args.xspan)
    time = np.linspace(start=0, stop=1000, num = args.tspan)
    aset = args.aspan


    save_path_head = Path(args.savepath)

    function_name = advection_diffusion_zero_visc.__name__

    save_path_folder = save_path_head / function_name

    save_path_folder.mkdir(exist_ok=True)

    for a in tqdm(range(aset), ncols=100, desc='Data Generation'):

        c = 100*np.abs(np.random.uniform(size=(1)))+2

        v = 20*np.abs(np.random.uniform(size=(1)))+2

        tseries = generate_time_series(time, x, c, v, advection_diffusion_zero_visc)

        id_string = f'vc_{v[0]/c[0]}_xspan_{args.xspan}_tspan_{args.tspan}'

        np.save(save_path_folder / id_string, tseries)


        




    



    