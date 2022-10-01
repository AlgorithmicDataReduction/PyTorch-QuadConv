from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm

import numpy as np

def generate_time_series(time,x,c,f_0,f):

    tseries = np.zeros((len(time),len(x)))

    for i,t in enumerate(time):

        tseries[i,] = f(t,x,c,f_0)

    return tseries

def advection_diffusion_zero_visc(t , x , c , v = 0.05 , terms = 50):
    '''
    Closed form solution for u(x,0) = -sin(pi*x) and zero boundary condtions
    '''

    sol =  np.zeros_like(x)

    for p in range(terms):

        sol +=  (-1)**p * (2*p*np.sin(np.pi*p*x) + (2*p+1)*np.cos((2*p+1)/2 * np.pi * x))

    mult_fac =  (v/c)**3 * np.exp((c/(2*v))*((x+1) - (c/2)*t))

    return mult_fac * sol


def simple_transport(t,x,c,f_0):

    sol = f_0(x-c*t)

    return sol



if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--tspan" , type=int , default=100 , help="temporal size")
    parser.add_argument("--xspan" , type=int , default=100 , help="spatial size")
    parser.add_argument("--aspan" , type=int , default=100 , help="initial value")
    parser.add_argument("--savepath", type=str, default='./' , help='save path for the data')

    args, _ = parser.parse_known_args()

    x = np.linspace(start=-1, stop=1, num = args.xspan)
    time = np.linspace(start=0, stop=10, num = args.tspan)
    aset = args.aspan


    save_path_head = Path(args.savepath)

    function_name = simple_transport.__name__

    save_path_folder = save_path_head / function_name

    save_path_folder.mkdir(exist_ok=True)

    for a in tqdm(range(aset), ncols=100, desc='Data Generation'):

        c = 0.001 * np.abs(np.random.uniform(size=(1))) + 0.25

        s = 0.001 * np.abs(np.random.uniform(size=(1))) + 0.25

        amp = 0.001 * np.abs(np.random.uniform(size=(1))) + 1

        f_0 =  lambda x : amp * np.exp(-0.5*((x+0.75)/s)**2)

        tseries = generate_time_series(time, x, c, f_0, simple_transport)

        id_string = f'c_{c[0]}_xspan_{args.xspan}_tspan_{args.tspan}'

        tseries = tseries.reshape(args.tspan, args.xspan, 1)

        np.save(save_path_folder / id_string, tseries)
