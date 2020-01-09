import lattice
import thermodynamics

import numpy as np
import matplotlib.pyplot as plt



"Testing lattice module"

#Initial = lattice.Initialise_Random_State(10,1.0,1.0,type='2d')
#Initial.Renormalise(2)
#Initial.Display()
#Initial.animate(100,50,save=False)

"Testing thermodynamics module"

#thermodynamics.Energy_Magnetization(20,2,1,1000,type='2d')

#state,E,M = thermodynamics.Converged_E_M(500,1,1,50,5)
#print("E = ",E,", M = ",M,"\n Spins = ",Initial.spin)

'''critical exp fitting'''
# Want to choose 'length' to avoid autocorrelated data
thermodynamics.SHeat_Susept(10,1,3,5,5000,50,1,type='2d')
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

'''make training data'''
#data = thermodynamics.Training_Data(5,1,1,1000,100,type='2d')
#np.savetxt('test_training_data.txt',data)

#for temp in np.linspace(1,3,10):
    #data = thermodynamics.Training_Data(20,temp,1,10000,100,type='2d')
    #np.savetxt('training_data_temp%s.txt'%temp,data)

'''using rbm'''
#import rbm
#for temp in np.linspace(1,3,10):
    #vis = 20**2
    #hid = (20**2)
    #r = rbm.RBM(num_visible = vis, num_hidden = hid)
    #training_data = np.loadtxt('training_data_temp%s.txt'%temp)
    #r.train(training_data)
    #data = r.daydream(10)
    #print(data)
    #np.savetxt('generated_data_temp%s.txt'%temp,data)


'''test generated data'''
#test_data = []

#for temp in np.linspace(1,3,10)[:7]:
    #data = np.loadtxt('generated_data_temp%s.txt'%temp)
    #M, E, susept, specific_heat = thermodynamics.Test_Generated(data,5,1.0,1.0,type='2d')
    #test_data.append((temp,M,E,susept,specific_heat))
    
#np.savetxt('results_data_testing.txt',test_data)

#print(np.loadtxt('generated_data_temp1.0.txt'))

#fig,ax=plt.subplots()
#plt.plot(test_data[:][0],test_data[:][1])
#plt.show()
