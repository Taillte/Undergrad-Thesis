import lattice
import thermodynamics
import rbm

import numpy as np
import matplotlib.pyplot as plt



"Testing lattice module"

#Initial = lattice.Initialise_Random_State(20,1.0,1.0,type='2d')
#for i in range(1000):
    #Initial.metropolis()
#r = rbm.RBM(num_visible = int(20**2), num_hidden = int((20**2)/2))
#training_data = thermodynamics.Training_Data(20,1.0,1.0,1000,1000,type='2d')
#np.savetxt('example_training_data_t=1.0.txt',training_data)
#training_data = np.loadtxt('example_training_data_t=1.0.txt')
#r.train(training_data,epochs=int(1000),learning_rate=0.1,batch_size=100,k=1)
#generated = r.daydream(1000,training_data[0])
#Initial.set_spins(generated[500])
#Initial.Renormalise(2)
#Initial.Display()
#Initial.animate(100,50,save=False)

"Testing thermodynamics module"

#thermodynamics.Energy_Magnetization(20,2,1,1000,type='2d')
#thermodynamics.SHeat_Susept(100,0,1,20,1000,100,1.0,type='1d',plot=True,analytic_compare=True)

#state,E,M = thermodynamics.Converged_E_M(500,1,1,50,5)
#print("E = ",E,", M = ",M,"\n Spins = ",Initial.spin)

'''critical exp fitting'''
# Want to choose 'length' to avoid autocorrelated data
#thermodynamics.SHeat_Susept(100,0,1,10,1000,5,1,type='1d',plot=False)
#test_data = np.loadtxt('generated_data_thermodynamics.txt')

#points = np.loadtxt('magnetization3.csv')
#points = points[points[:,0]<2.3]
#points = points[points[:,0]>1]
#np.savetxt('mag2_Tave_trimmed_Tc.txt',points,delimiter=',')
#fit = thermodynamics.Convex_fit(points)
#fit = fit[fit[:,1]>0.5]
#np.savetxt("convex_hull3 _Tc.txt",fit,delimiter=',')

#state, specific_heat, susceptibitily = thermodynamics.Converged_SHeat_Susept(500,1,1,100,400,type='1d')
#print("SH = ",specific_heat,", S = ",susceptibitily,"\n Spins = ",state.spin)

#thermodynamics.SHeat_Susept(10, 1.0, 5.0, 10.0, 2000, 500, 1.0, type='2d')
#thermodynamics.SHeat_Susept(500, 1.0, 5.0, 10.0, 1000, 500, 1.0, type='1d')

thermodynamics.compare_1d_rg(200,0,1,20,1000,50,1)

'''make training data'''
#data = thermodynamics.Training_Data(10,1,1,1000,100,type='2d')
#np.savetxt('test_training_data.txt',data)

#for temp in np.linspace(0.0001,1,10):
    #data = thermodynamics.Training_Data(100,temp,1,1000,100,type='1d')
    #np.savetxt('training_data_temp%s.txt'%temp,data)

'''using rbm'''
#import rbm
#for temp in np.linspace(0.0001,1,10):
    #vis = 100
    #hid = 100
    #r = rbm.RBM(num_visible = vis, num_hidden = hid)
    #training_data = np.loadtxt('training_data_temp%s.txt'%temp)
    #r.train(training_data)
    #data = r.daydream(100)
    #print(data)
    #np.savetxt('generated_data_temp%s.txt'%temp,data)


'''test generated data'''
#test_data = np.zeros((5,10))
#index = 0

#for temp in np.linspace(0.0001,1,10):
    #data = np.loadtxt('generated_data_temp%s.txt'%temp)
    #M, E, susept, specific_heat = thermodynamics.Test_Generated(data,100,temp,1.0,type='1d')
    #test_data[0][index] = temp
    #test_data[1][index] = M
    #test_data[2][index] = E
    #test_data[3][index] = susept
    #test_data[4][index] = specific_heat
    #index +=1
 
#print(test_data.shape)
#np.savetxt('results_data_testing.txt',test_data)

#print(np.loadtxt('generated_data_temp1.0.txt'))
#print(test_data[0])

#fig,ax=plt.subplots()
#plt.plot(test_data[0],test_data[1])
#plt.show()
#50,1.0,1000,5000,1000,0,1,10,type='1d'

''' finding hyperparameters'''
'''1d'''
#print(thermodynamics.Generate_and_Test(100,1.0,2000,1000,1000,0,1,10,type='1d',gen_training=False,epochs=200))
#thermodynamics.Compare_RMB_Params_Temp(100,1.0,2000,1000,1000,0,1,10,type='1d')
#epochs = np.loadtxt('optimal_epochs_1d.txt')
#thermodynamics.Generate_and_Test(100,1.0,2000,1000,1000,0,1,10,type='1d',gen_training=False,epochs=5000)
#thermodynamics.Autocorrelations_in_generated(100,1.0,80,100,0.88889,type='1d',plot=True,epochs=1000,stacked=False,k=1)

#thermodynamics.KL_Div(100,0.88889,type='1d',epochs=2000,plot=True,k=1)
#epochs,errors1 = thermodynamics.KL_Div(100,0.66667,type='1d',epochs=2000,plot=False,k=1)
#epochs,errors2 = thermodynamics.KL_Div(100,0.66667,type='1d',epochs=2000,plot=False,k=15)
#thermodynamics.Compare_KL_Div(epochs,errors1,errors2)

'''1d - weight matrices'''
#temperatures = np.linspace(0,1,10)
#thermodynamics.Compare_Weights(size,ratios,epochs,temperatures,type='1d')

#data = np.loadtxt('training_data_temp0.5555599999999999.txt')
#thermodynamics.Weight_Matrix(100,0.16,15000,data)
#plt.show()
# Running for longer with ratio=2 -> visible correlations become diagonal
'''2d'''
#thermodynamics.Generate_and_Test(10,1.0,2000,1000,1000,1,3,10,type='2d',gen_training=False,epochs=200)
#thermodynamics.Compare_RMB_Params_Temp(10,1.0,2000,1000,1000,1,3,10,type='2d',stacked=True)
#epochs = np.loadtxt('optimal_epochs_2d.txt')
#thermodynamics.Generate_and_Test(10,1.0,2000,1000,1000,1,3,10,type='2d',gen_training=False,epochs=epochs)

#epochs,errors1 = thermodynamics.KL_Div(10,2.7777777777777777,type='2d',epochs=2000,plot=False,k=1)
#epochs,errors2 = thermodynamics.KL_Div(10,2.7777777777777777,type='2d',epochs=2000,plot=False,k=1,stacked=True)
#thermodynamics.Compare_KL_Div(epochs,errors1,errors2)

#thermodynamics.Autocorrelations_in_generated(10,1.0,170,100,2.7777777777777777,type='2d',plot=True,epochs=1000,stacked=False,k=1)


'''3d'''
#thermodynamics.Generate_and_Test(6,1.0,1000,1000,1000,3.5,5.5,10,type='3d',gen_training=False,epochs=200)
#thermodynamics.Compare_RMB_Params_Temp(6,1.0,1000,1000,1000,3.5,5.5,10,type='3d',stacked=True)
#epochs = np.loadtxt('optimal_epochs_3d.txt')
#print(epochs.shape)
#thermodynamics.Generate_and_Test(6,1.0,1000,1000,1000,3.5,5.5,10,type='3d',gen_training=False,epochs=200,stacked=True)

#epochs,errors1 = thermodynamics.KL_Div(6,5.055555555555555,type='3d',epochs=2000,plot=False,k=1)
#epochs,errors2 = thermodynamics.KL_Div(6,5.055555555555555,type='3d',epochs=2000,plot=False,k=15,stacked=False)
#thermodynamics.Compare_KL_Div(epochs,errors1,errors2)

#thermodynamics.Autocorrelations_in_generated(6,1.0,140,100,5.055555555555555,type='3d',plot=True,epochs=1000,stacked=False,k=1)


'''compare tile errors for two datasets'''
# Can't save 3d array...

'''run different conditions and save'''
#X, Y, mag_errors, energy_errors, suscept_errors, SHeat_errors = thermodynamics.Compare_RMB_Params_Temp(6,1.0,2000,1000,1000,3.5,5.5,10,num_epochs=5,type='3d',stacked=True,k=1,plot=True)
#np.savetxt('X_3d.txt',X)
#np.savetxt('Y_3d.txt',Y)
#np.savetxt('tile_errs_3d_cd1_stacked_mag.txt',mag_errors)
#np.savetxt('tile_errs_3d_cd1_stacked_energy.txt',energy_errors)
#np.savetxt('tile_errs_3d_cd1_stacked_suscept.txt',suscept_errors)
#np.savetxt('tile_errs_3d_cd1_stacked_sh.txt',SHeat_errors)
'''load back in / run and compare specific graphs'''
#X = np.loadtxt('X_3d.txt')
#Y = np.loadtxt('Y_3d.txt')
#sample1 = np.loadtxt('tile_errs_3d_cd15_energy.txt')
#sample2 = np.loadtxt('tile_errs_3d_cd1_energy.txt')
#thermodynamics.Compare_Tile_Errors(X,Y,sample1,sample2,'Energy')

#X, Y, mag_errors, energy_errors, suscept_errors, SHeat_errors = thermodynamics.Compare_RMB_Params_Temp(100,1.0,2000,1000,1000,0,1,10,num_epochs=5,type='1d',plot=False)
#X2, Y2, mag_errors2, energy_errors2, suscept_errors2, SHeat_errors2 = thermodynamics.Compare_RMB_Params_Temp(100,1.0,2000,1000,1000,0,1,10,num_epochs=5,type='1d',k=3,plot=False)
# To change axis labels etc. go into thermodynamics module
#thermodynamics.Compare_Tile_Errors(X,Y,mag_errors,mag_errors2,'Magnetization')
#thermodynamics.Compare_Tile_Errors(X,Y,energy_errors,energy_errors2,'Energy')
#thermodynamics.Compare_Tile_Errors(X,Y,suscept_errors,suscept_errors2,'Susceptibility')
#thermodynamics.Compare_Tile_Errors(X,Y,SHeat_errors,SHeat_errors2,'Specific Heat')
#plt.show()



# NOTE: can plot divergence of mc stats from rbm stats
# some problem leading to flat rbm graph?


