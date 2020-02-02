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
#thermodynamics.SHeat_Susept(500, 1.0, 5.0, 10.0, 1000, 500, 1.0, type='÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷≥1d')

#thermodynamics.compare_1d_rg(500,0,1,20,5000,50,1)

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
#print(thermodynamics.Generate_and_Test(100,1.0,2000,1000,1000,0,1,10,type='1d',gen_training=False,epochs=200))
thermodynamics.Compare_RMB_Params_Temp(100,1.0,2000,1000,1000,0,1,10,type='1d')

#NOTE: can plot divergence of mc stats from rbm stats
"some problem leading to flat rbm graph"

" 1000 spin configurations for each of 25 different temperatures T = 0, 0.25, . . . , 6. Then the index A runs from 1 to N = 25000. In some cases, as we will see in Sec. 3.2, we use only a restricted set of configurations at high or low temperatures, then the index runs A = 1, . . . , N = 1000×(number of temperatures). We repeat the renewal procedure (23) many times (5000 epochs) - 13"
