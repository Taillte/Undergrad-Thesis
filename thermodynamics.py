#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as spc
import random
#from matplotlib.animation import ArtistAnimation
#%matplotlib qt5
from scipy.ndimage import gaussian_filter

import lattice

# Take in generated data and do stats on it
# First sample is a random initialisation
def Test_Generated(data,size,temp,K,type=type):
    (num_samples,num_sites)=data.shape
    mean_M=0
    mean_E=0
    mean_M_squared=0
    mean_E_squared=0
    
    # Should only take one in every few.. correlations between adjacent values
    for i in range(num_samples-1):
        Initial = lattice.Initialise_Random_State(size,temp,K,type=type)
        Initial.set_spins(data[i+1])
        m = np.abs(Initial.magnetisation())
        print(m)
        en = Initial.energy()
        mean_M += m
        mean_M_squared += m**2
        mean_E += en
        mean_E_squared += en**2
        
    mean_M = mean_M /float(num_samples-1)
    mean_M_squared = mean_M_squared /float(num_samples-1)
     
    susept = np.abs(mean_M_squared - mean_M**2)/(temp)
     
    mean_E = mean_E /float(num_samples-1)
    mean_E_squared = mean_E_squared /float(num_samples-1)
     
    specific_heat = np.abs(mean_E_squared - mean_E**2)/(temp **2)
    
    return mean_M, mean_E, susept, specific_heat

        

# Show convergence of Energy & Magnetisation with iterations
def Energy_Magnetization(size, temp, K, length, type='1d',show=True):
    num = []
    final_E = []
    final_M = []

    Initial = lattice.Initialise_Random_State(size,temp,K,type=type)
    for i in range(length):
        Initial.metropolis()
        num.append(i)
        final_E.append(Initial.energy())
        final_M.append(np.abs(Initial.magnetisation()))
    
    if show==True:
        fig, ax = plt.subplots()
        plt.plot(num, final_E, marker = '.', label='Energy')
        plt.legend()
        plt.xlabel('Iterations of Metropolis Algorithm')
        plt.ylabel('Average Energy per cell')

        fig, ax = plt.subplots()
        plt.plot(num, final_M, marker = '.', label='Mag')
        plt.legend()
        plt.xlabel('Iterations of Metropolis Algorithm')
        plt.ylabel('Average Magnetisation per cell')
            
        plt.show()
        
        
# Returns lattices for given conditions ready for rbm input
def Training_Data(size,temp,K,length,number,type='1d'):
    if type=='1d':
        data = np.zeros((number,size))
    if type=='2d':
        data = np.zeros((number,size**2))
    if type=='3d':
        data = np.zeros((number,size**3))
        
    for i in range(length):
        Initial.metropolis()
        
    for num in range(number):
        Initial = lattice.Initialise_Random_State(size,temp,K,type=type)
        for i in range(length):
            Initial.metropolis()
        if type=='1d':
            spins = np.reshape(Initial.get_spins(),size)
            for i in range(size):
                if (spins[i]==-1.0):
                    spins[i] = 0
        elif type=='2d':
            spins = np.reshape(Initial.get_spins(),size**2)
            for i in range(size**2):
                if (spins[i]==-1.0):
                    spins[i] = 0
        elif type=='3d':
            spins = np.reshape(Initial.get_spins(),size**3)
            for i in range(size**3):
                if (spins[i]==-1.0):
                    spins[i] = 0
        data[num,:] = spins[:]

    return data
    


    
# Find converged state, energy and magnetization for a given lattice
def Converged_E_M(size, temp, K, length, samples, type='1d'):
    final_E = []
    final_M = []
    if temp == 0:
        temp = 0.00001
    Initial = lattice.Initialise_Random_State(size,temp,K,type=type)
    for i in range(length):
        Initial.metropolis()
    for sample in range(samples):
        for i in range(length):
            Initial.metropolis()
        final_E.append(Initial.energy())
        final_M.append(np.abs(Initial.magnetisation()))
        
    E = np.sum(final_E)/samples
    M = np.sum(final_M)/samples
    
    return Initial,E,M


def Convex_fit(points):
    from scipy.spatial import ConvexHull, convex_hull_plot_2d
    hull = ConvexHull(points)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(points[:,0], points[:,1], 'o')
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
    plt.figure()
    plt.plot(points[hull.vertices, 0],points[hull.vertices, 1],'ro')
    fit = points[hull.vertices,:]

    plt.show()
    return fit
        


# Shows dependence of E and M on T and calculates Tc
# Added averaging over 10 trials for each temperature
def Energy_Magnetization_Temp(size, min_temp, max_temp, number_temps, K, length,type = '1d'):
    final_T = []
    final_E = []
    final_M = []
    
    temperatures = np.log(np.linspace(np.exp(min_temp), np.exp(max_temp), number_temps))
    
    for temp in temperatures:
        n_trials = 10
        energies = np.zeros(n_trials)
        mags = np.zeros(n_trials)
        for n in range(n_trials):
            if temp == 0:
                temp = 0.0001
            Initial = lattice.Initialise_Random_State(size,temp,K,type=type)
            for i in range(length):
                Initial.metropolis()
            energies[n]=Initial.energy()
            mags[n]=np.abs(Initial.magnetisation())
            
        final_T.append(temp)
        final_E.append(np.median(energies))
        final_M.append(np.median(mags))
        print(temp)


    smoothed_e = gaussian_filter(final_E, 3.)
    smoothed_m = gaussian_filter(final_M,3.)

    dx = np.diff(final_T)
    dy_e = np.diff(smoothed_e)
    dy_m = np.diff(smoothed_m)

    slope_e = (dy_e/dx)
    slope_m = (dy_m/dx)

    max_slope_e = (max(slope_e))
    min_slope_m = (min(slope_m))

    Curie_e = 0.0
    Curie_m = 0.0

    for i in range(len(final_T)-1):
        if slope_e[i]==max_slope_e:
            Curie_e = final_T[i]
        if slope_m[i]==min_slope_m:
            Curie_m = final_T[i]

    print('Curie temperature from energy is', Curie_e)
    print('Curie temperature from magnetisation is', Curie_m)


    mag_data = np.concatenate((np.array([final_T]).T,np.array([final_M]).T),axis=1)
    np.savetxt("magnetization3.csv",mag_data)
    
    fig, ax = plt.subplots()
    plt.plot(final_T, final_E, '.', label='Energy-Temp')
    plt.legend()
    plt.xlabel('Fundamental Temperature')
    plt.ylabel('Average Energy per cell after Convergence')

    fig, ax = plt.subplots()
    plt.plot(final_T, final_M, '.', label='Mag-Temp')
    plt.legend()
    plt.xlabel('Fundamental Temperature')
    plt.ylabel('Average Magnetisation per cell after Convergence')

    plt.show()
    
    
# Find converged sheat and suscept for given lattice
def Converged_SHeat_Susept(size,temp,K,length,samples,type='1d'):

    Initial = lattice.Initialise_Random_State(size,temp,K,type=type)
    
    mean_M = 0.0
    mean_M_squared = 0.0
    mean_E = 0.0
    mean_E_squared = 0.0
    const = 0
    
    if temp == 0:
        temp = 0.001
    for i in range(length+samples):
        Initial.metropolis()
        
        if i > length:
            m = Initial.magnetisation()
            en = Initial.energy()
            mean_M += m
            mean_M_squared += m**2
            mean_E += en
            mean_E_squared += en**2
    
    
    mean_M = np.abs(mean_M /samples)
    mean_M_squared = mean_M_squared /samples
    
    susept = np.abs(mean_M_squared - mean_M**2)/(temp)
    
    mean_E = mean_E /samples
    mean_E_squared = mean_E_squared /samples
    
    specific_heat = np.abs(mean_E_squared - mean_E**2)/(temp **2)
    
    
    return Initial, specific_heat, susept




# plot magnetisation, energy, specific heat and susceptibility wrt temp
def SHeat_Susept(size, min_temp, max_temp, number_temps, length, samples, K, type='1d'):
    final_T = []
    final_M = []
    final_E = []
    final_X = []
    final_SH = []
    if min_temp == 0:
        min_temp = 0.0001
    temperatures = np.linspace(min_temp, max_temp, number_temps)
    for temp in temperatures:
        
        mean_M = 0.0
        mean_M_squared = 0.0
        mean_E = 0.0
        mean_E_squared = 0.0
        const = 0

        Initial = lattice.Initialise_Random_State(size,temp,K,type=type)
        for i in range(length+samples):
            Initial.metropolis()
            
            if i > length:
                for n in range(length):
                    Initial.metropolis()
                m = Initial.magnetisation()
                en = Initial.energy()
                mean_M += m
                mean_M_squared += m**2
                mean_E += en
                mean_E_squared += en**2
        
        
        mean_M = mean_M /float(samples)
        mean_M_squared = mean_M_squared /float(samples)
        
        susept = np.abs(mean_M_squared - mean_M**2)/(temp)
        
        mean_E = mean_E /float(samples)
        mean_E_squared = mean_E_squared /float(samples)
        
        specific_heat = np.abs(mean_E_squared - mean_E**2)/(temp **2)
        
        final_T.append(temp)
        final_M.append(mean_M)
        final_E.append(mean_E)
        final_X.append(susept)
        final_SH.append(specific_heat)
        print(temp)
    
    
    '''deriving Curie temp from M and E'''
    
    smoothed_e = gaussian_filter(final_E, 3.)
    smoothed_m = gaussian_filter(final_M,3.)

    dx = np.diff(final_T)
    dy_e = np.diff(smoothed_e)
    dy_m = np.diff(smoothed_m)

    slope_e = (dy_e/dx)
    slope_m = (dy_m/dx)

    max_slope_e = (max(slope_e))
    min_slope_m = (min(slope_m))

    Curie_e = 0.0
    Curie_m = 0.0

    for i in range(len(final_T)-1):
        if slope_e[i]==max_slope_e:
            Curie_e = final_T[i]
        if slope_m[i]==min_slope_m:
            Curie_m = final_T[i]

    print('Curie temperature from energy is', Curie_e)
    print('Curie temperature from magnetisation is', Curie_m)
    
    '''saving data and graphing'''
    
    mag_data = np.concatenate((np.array([final_T]).T,np.array([final_M]).T),axis=1)
    np.savetxt("magnetization3.csv",mag_data)
    
    fig, ax = plt.subplots()
    plt.plot(final_T, final_E, '.', label='Energy-Temp')
    plt.legend()
    plt.xlabel('Fundamental Temperature')
    plt.ylabel('Average Energy per cell after Convergence')

    fig, ax = plt.subplots()
    plt.plot(final_T, final_M, '.', label='Mag-Temp')
    plt.legend()
    plt.xlabel('Fundamental Temperature')
    plt.ylabel('Average Magnetisation per cell after Convergence')

    
    fig, ax = plt.subplots()
    plt.plot(final_T, final_X, '.', label='Suseptibility')
    plt.legend()
    plt.xlabel('Fundamental Temperature')
    plt.ylabel('Suseptibility')

    fig, ax = plt.subplots()
    plt.plot(final_T, final_SH, '.', label='Specific Heat')
    plt.legend()
    plt.xlabel('Fundamental Temperature')
    plt.ylabel('Specific Heat')


    plt.show()

'''heat capacity = standard deviation of E **2 = change in E between iterations? then divided by k times T **2'''








