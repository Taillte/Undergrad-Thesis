#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as spc
import random
#from matplotlib.animation import ArtistAnimation
#%matplotlib qt5
from scipy.ndimage import gaussian_filter

import lattice
import rbm

def Compare_RMB_Params(size,K,length,number_train,number_gen,min_temp,max_temp,number_temps,type='1d',min_epoch=100,max_epoch=10000,num_epochs=10):
    epochs = np.linspace(min_epoch,max_epoch,num_epochs)
    differences = np.zeros((4,num_epochs))
    for i in range(num_epochs):
        difference_for_epoch = Generate_and_Test(size,K,length,number_train,number_gen,min_temp,max_temp,number_temps,type='1d',plot=False,gen_training=False,epochs=int(epochs[i]))
        differences[0][i] = difference_for_epoch[0]
        differences[1][i] = difference_for_epoch[1]
        differences[2][i] = difference_for_epoch[2]
        differences[3][i] = difference_for_epoch[3]
    
    fig, ax = plt.subplots()
    plt.plot(epochs, differences[0], '.', label='magnetization')
    plt.plot(epochs, differences[1], '.', label='energy')
    plt.plot(epochs, differences[2], '.', label='susceptibility')
    plt.plot(epochs, differences[3], '.', label='specific heat')
    plt.legend()
    plt.ylim(0,1)
    plt.xlabel('Training Epochs')
    plt.ylabel('Difference between MC and RBM Stats')
    
    plt.show()
    
def Compare_RMB_Params_Temp(size,K,length,number_train,number_gen,min_temp,max_temp,number_temps,type='1d',min_epoch=10,max_epoch=1000,num_epochs=30):
    from mpl_toolkits import mplot3d

    epochs = np.linspace(min_epoch,max_epoch,num_epochs)
    if min_temp==0.0:
        min_temp=0.00001
    temperatures = np.linspace(min_temp,max_temp,number_temps)
    mag_errors = np.zeros((num_epochs,number_temps))
    energy_errors = np.zeros((num_epochs,number_temps))
    suscept_errors = np.zeros((num_epochs,number_temps))
    SHeat_errors = np.zeros((num_epochs,number_temps))
    #epochs_for_plotting = []
    #temperatures_for_plotting = []
    for i in range(num_epochs):
        difference_for_epoch = Generate_and_Test(size,K,length,number_train,number_gen,min_temp,max_temp,number_temps,type='1d',plot=False,gen_training=False,epochs=int(epochs[i]))

        mag_errors[i] = difference_for_epoch[0]
        energy_errors[i] = difference_for_epoch[1]
        suscept_errors[i] = difference_for_epoch[2]
        SHeat_errors[i] = difference_for_epoch[3]

    X, Y = np.meshgrid(epochs,temperatures)
    #print('X is ',X)
    #print('Y is ',Y)
    X = X.T
    Y = Y.T

    #X,Y = epochs,temperatures
    #print(X.shape, Y.shape,mag_errors.shape)
    
    z_min, z_max = -np.abs(mag_errors).max(), np.abs(mag_errors).max()
    fig, ax = plt.subplots()
    c = ax.pcolormesh(X, Y, mag_errors, cmap='RdBu', vmin=z_min, vmax=z_max)
    ax.set_title('mag errors')
    # set the limits of the plot to the limits of the data
    ax.axis([X.min(), X.max(), Y.min(), Y.max()])
    ax.set_xlabel('epochs')
    ax.set_ylabel('temperature')
    fig.colorbar(c, ax=ax)
    
    
    z_min, z_max = -np.abs(energy_errors).max(), np.abs(energy_errors).max()
    fig, ax = plt.subplots()
    c = ax.pcolormesh(X, Y, energy_errors, cmap='RdBu', vmin=z_min, vmax=z_max)
    ax.set_title('energy errors')
    # set the limits of the plot to the limits of the data
    ax.axis([X.min(), X.max(), Y.min(), Y.max()])
    ax.set_xlabel('epochs')
    ax.set_ylabel('temperature')
    fig.colorbar(c, ax=ax)
    
    z_min, z_max = -np.abs(suscept_errors).max(), np.abs(suscept_errors).max()
    fig, ax = plt.subplots()
    c = ax.pcolormesh(X, Y, suscept_errors, cmap='RdBu', vmin=z_min, vmax=z_max)
    ax.set_title('suscept errors')
    # set the limits of the plot to the limits of the data
    ax.axis([X.min(), X.max(), Y.min(), Y.max()])
    ax.set_xlabel('epochs')
    ax.set_ylabel('temperature')
    fig.colorbar(c, ax=ax)
    
       
    z_min, z_max = -np.abs(SHeat_errors).max(), np.abs(SHeat_errors).max()
    fig, ax = plt.subplots()
    c = ax.pcolormesh(X, Y, SHeat_errors, cmap='RdBu', vmin=z_min, vmax=z_max)
    ax.set_title('SHeat errors')
    # set the limits of the plot to the limits of the data
    ax.axis([X.min(), X.max(), Y.min(), Y.max()])
    ax.set_xlabel('epochs')
    ax.set_ylabel('temperature')
    fig.colorbar(c, ax=ax)
    

    plt.show()
    

#currently: num_hidden = num_visible -> add variable in def 'ratio_hidden'
def Generate_and_Test(size,K,length,number_train,number_gen,min_temp,max_temp,number_temps,type='1d',plot=True,gen_training=True,epochs=100):
    if min_temp==0.0:
        min_temp=0.00001
    
    temperatures = np.linspace(min_temp,max_temp,number_temps)
    
    training_stats=np.zeros((5,number_temps))

    training_data=[]
    if gen_training==True:
        for i in range(number_temps):
            #print(Training_Data(size,temperatures[i],K,length,number_train,type=type))
            training_data.append(Training_Data(size,temperatures[i],K,length,number_train,type=type))
            np.savetxt('training_data_temp%s.txt'%temperatures[i],training_data[i])

    #np.savetxt('training_data.txt',training_data)
    
    ratio_hidden = 0.5
    autocorrelation_guess_rbm = 100
    #print(training_data)

    gen_data=[]
    for i in range(number_temps):
        if type=='1d':
            vis = size
            hid = int(vis*ratio_hidden)
        if type=='2d':
            vis = size**2
        if type=='3d':
            vis = size**3
        hid = int(vis*ratio_hidden)
        r = rbm.RBM(num_visible = vis, num_hidden = hid)
        #print(training_data[i])
        if gen_training==True:
            r.train(training_data[i],epochs=epochs,learning_rate=0.1,batch_size=100)
            gen_data.append(r.daydream(number_gen*autocorrelation_guess_rbm,training_data[i][0]))

        else:
            training_data_sample = np.loadtxt('training_data_temp%s.txt'%temperatures[i])
            r.train(training_data_sample,epochs=epochs,learning_rate=0.1,batch_size=100)
            gen_data.append(r.daydream(number_gen*autocorrelation_guess_rbm,training_data_sample[0]))
            training_stats[0][i]=temperatures[i]
            training_stats[1][i], training_stats[2][i], training_stats[3][i], training_stats[4][i] = Test_Generated(training_data_sample,size,temperatures[i],K,type=type)
        #np.savetxt('generated_data_temp%s.txt'%temperatures[i],gen_data[i])
        
    test_data = np.zeros((5,number_temps))

    for i in range(number_temps):
        M, E, susept, specific_heat = Test_Generated(gen_data[i],size,temperatures[i],K,type=type,autocorrelation=autocorrelation_guess_rbm)
        test_data[0][i] = temperatures[i]
        test_data[1][i] = M
        test_data[2][i] = E
        test_data[3][i] = susept
        test_data[4][i] = specific_heat
        if gen_training==True:
            training_stats[1][i], training_stats[2][i], training_stats[3][i], training_stats[4][i] = Test_Generated(training_data[i],size,temperatures[i],K,type=type)
        
    # Want to compare stats of MC and RBM data
    differences = np.zeros((5,number_temps))
    differences[0]=np.abs(training_stats[1] - test_data[1])
    differences[1]=np.abs(training_stats[2] - test_data[2])
    differences[2]=np.abs(training_stats[3] - test_data[3])
    differences[3]=np.abs(training_stats[4] - test_data[4])
        
    
        
    if plot==True:
    
        fig, ax = plt.subplots()
        plt.plot(temperatures, differences[0], '.', label='magnetization')
        plt.plot(temperatures, differences[1], '.', label='energy')
        plt.plot(temperatures, differences[2], '.', label='susceptibility')
        plt.plot(temperatures, differences[3], '.', label='specific heat')
        plt.legend()
        plt.ylim(0,1)
        plt.xlabel('Fundamental Temperature')
        plt.ylabel('Difference between MC and RBM Stats')
        
        fig, ax = plt.subplots()
        plt.plot(training_stats[0], training_stats[2], '.', label='MC')
        plt.plot(test_data[0], test_data[2], '.', label='rbm data')
        plt.legend()
        plt.xlabel('Fundamental Temperature')
        plt.ylabel('Average Energy per cell after Convergence')

        fig, ax = plt.subplots()
        plt.plot(training_stats[0], training_stats[1], '.', label='MC')
        plt.plot(test_data[0], test_data[1], '.', label='rbm data')
        plt.legend()
        plt.xlabel('Fundamental Temperature')
        plt.ylabel('Average Magnetisation per cell after Convergence')
         
        fig, ax = plt.subplots()
        plt.plot(training_stats[0], training_stats[3], '.', label='MC')
        plt.plot(test_data[0], test_data[3], '.', label='rbm data')
        plt.legend()
        plt.xlabel('Fundamental Temperature')
        plt.ylabel('Suseptibility')

        fig, ax = plt.subplots()
        plt.plot(training_stats[0], training_stats[4], '.', label='MC')
        plt.plot(test_data[0], test_data[4], '.', label='rbm data')
        plt.legend()
        plt.xlabel('Fundamental Temperature')
        plt.ylabel('Specific Heat')

        plt.show()
        
    return differences
    #return np.sum(differences,axis=1)/number_temps
    #np.savetxt('generated_data_thermodynamics.txt',test_data)
     


# Take in generated data and do stats on it
# First sample is a random initialisation
def Test_Generated(data,size,temp,K,type=type,autocorrelation=1):
    (num_samples,num_sites)=data.shape
    num_samples=int(num_samples/autocorrelation)
    mean_M=0
    mean_E=0
    mean_M_squared=0
    mean_E_squared=0
    
    # Should only take one in every few.. correlations between adjacent values = ?
    for i in range(num_samples*autocorrelation):
        if (i%autocorrelation==0):
            Initial = lattice.Initialise_Random_State(size,temp,K,type=type)
            Initial.set_spins(data[i])
            m = np.abs(Initial.magnetisation())
            en = Initial.energy()
            mean_M += m
            mean_M_squared += m**2
            mean_E += en
            mean_E_squared += en**2
        
    mean_M = mean_M /float(num_samples)
    mean_M_squared = mean_M_squared /float(num_samples)
     
    susept = np.abs(mean_M_squared - mean_M**2)/(temp)
     
    mean_E = mean_E /float(num_samples-1)
    mean_E_squared = mean_E_squared /float(num_samples-1)
     
    specific_heat = np.abs(mean_E_squared - mean_E**2)/(temp **2)
    
    return mean_M, mean_E, susept, specific_heat


def compare_1d_rg(size, min_temp, max_temp, number_temps, length, samples, K):
    final_T,final_E,final_M,final_X,final_SH = SHeat_Susept(size, min_temp, max_temp, number_temps, length, samples, K,type='1d',plot=False)
    rgfinal_T,rgfinal_E,rgfinal_M,rgfinal_X,rgfinal_SH = SHeat_Susept(int(size/2.0), min_temp, max_temp, number_temps, length, samples, 0.5 * np.log(np.cosh(2*K)),type='1d',plot=False)

    fig, ax = plt.subplots()
    plt.plot(final_T, final_E, '.', label='Energy-Temp')
    plt.plot(rgfinal_T, rgfinal_E, '.', label='one rg step')
    plt.legend()
    plt.xlabel('Fundamental Temperature')
    plt.ylabel('Average Energy per cell after Convergence')

    fig, ax = plt.subplots()
    plt.plot(final_T, final_M, '.', label='Mag-Temp')
    plt.plot(rgfinal_T, rgfinal_M, '.', label='one rg step')
    plt.legend()
    plt.xlabel('Fundamental Temperature')
    plt.ylabel('Average Magnetisation per cell after Convergence')
     
    fig, ax = plt.subplots()
    plt.plot(final_T, final_X, '.', label='Suseptibility')
    plt.plot(rgfinal_T, rgfinal_X, '.', label='one rg step')
    plt.legend()
    plt.xlabel('Fundamental Temperature')
    plt.ylabel('Suseptibility')

    fig, ax = plt.subplots()
    plt.plot(final_T, final_SH, '.', label='Specific Heat')
    plt.plot(rgfinal_T, rgfinal_SH, '.', label='one rg step')
    plt.legend()
    plt.xlabel('Fundamental Temperature')
    plt.ylabel('Specific Heat')

    plt.show()
    
        

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
# ** Should check thermodynamics of this data
def Training_Data(size,temp,K,length,number,type='1d'):
    if type=='1d':
        data = np.zeros((number,size))
    if type=='2d':
        data = np.zeros((number,size**2))
    if type=='3d':
        data = np.zeros((number,size**3))

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
    
    print(data)
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
def SHeat_Susept(size, min_temp, max_temp, number_temps, length, samples, K, type='1d',plot=True):
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
                mean_M += np.abs(m)
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
    
    if plot==True:
        
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
        
    return final_T,final_E,final_M,final_X,final_SH

'''heat capacity = standard deviation of E **2 = change in E between iterations? then divided by k times T **2'''








