#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as spc
import random
from matplotlib.animation import ArtistAnimation
from matplotlib.animation import FuncAnimation, PillowWriter
#from mpl_toolkits import mplot3d

class state:

    def __init__(self,size,K,T,type='1d'):
        self.size = size
        self.K = K
        self.T = T
        self.beta = (T)**(-1.0)
        self.list = []
        self.type = type
        #* spc.Boltzmann

        
        if type == '1d':
            self.points = np.array([[a] for a in np.linspace(0,size-1,size)])
            self.spin = np.zeros((size))


        if type == '2d':
            self.points = np.array([[a, b] for a in np.linspace(0,size-1,size) for b in np.linspace(0,size-1,size)])
            self.spin = np.zeros((size,size))
            
        if type == '3d':
            self.points = np.array([[a, b,c] for a in np.linspace(0,size-1,size) for b in np.linspace(0,size-1,size) for c in np.linspace(0,size-1,size)])
            self.spin = np.zeros((size,size,size))
            
        self.renorm_size=self.size
        self.renorm_spin=self.spin
        self.renorm_points=self.points
        self.renorm_K=0.0

    
    def get_size(self):
        return self.size
        
    def set_spins(self,spins):
        #print(spins.shape[0],spins.shape[0]**(1.0/3.0))
        for i in range(spins.shape[0]):
            if spins[i] == 0:
                spins[i] = -1
                
        if self.type=='1d':
            self.spin = spins
        if self.type=='2d':
            self.spin = np.reshape(spins,(int(spins.shape[0]**0.5),int(spins.shape[0]**0.5)))
            self.renorm_spin = self.spin
        if self.type=='3d':
            self.spin = np.reshape(spins,(int(round(spins.shape[0]**(1.0/3.0))),int(round(spins.shape[0]**(1.0/3.0))),int(round(spins.shape[0]**(1.0/3.0)))))

    
    def get_coupling(self):
        return K
        
    def get_temp(self):
        return T
    
    def get_spins(self):
        return self.spin
        
    def get_type(self):
        return self.type
    
    def energy(self):
     
        interaction = 0.0
        
        if self.type == '3d':
            for i in range(self.size):
                for j in range(self.size):
                    for k in range(self.size):
                        interaction+= self.energy_cell_for_energy(i,j,k)
            return interaction/(self.size**3)

        if self.type == '2d':
            for i in range(self.size):
                for j in range(self.size):
                    interaction+= self.energy_cell_for_energy(i,j,0)
            return interaction/(self.size**2)


        elif self.type == '1d':
            for i in range(self.size):
                interaction+= self.energy_cell_for_energy(i,0,0)
            return interaction/(self.size)

    '''returns average energy per element'''



    def energy_cell_for_energy(self,i,j,k):
        
        if self.type == '1d':
            if i == 0:
                left_spin = self.spin[self.size-1]
            else:
                left_spin = self.spin[i-1]
                
            return -self.spin[i] * self.K * left_spin

            

        if self.type == '2d':
            if i == 0:
                left_spin = self.spin[self.size-1][j]
            else:
                left_spin = self.spin[i-1][j]
                
            if j == 0:
                top_spin = self.spin[i][self.size-1]
            else:
                top_spin = self.spin[i][j-1]
        
            return -self.spin[i][j] * self.K * (left_spin + top_spin)
            

        if self.type=='3d':
        
            if i == 0:
                left_spin = self.spin[self.size-1][j][k]
            else:
                left_spin = self.spin[i-1][j][k]
            
            if i == (self.size-1):
                right_spin = self.spin[0][j][k]
            else:
                right_spin = self.spin[i+1][j][k]
            
            if j == 0:
                top_spin = self.spin[i][self.size-1][k]
            else:
                top_spin = self.spin[i][j-1][k]
                
            if j == (self.size-1):
                bottom_spin = self.spin[i][0][k]
            else:
                bottom_spin = self.spin[i][j+1][k]
                    
            if k==0:
                front_spin = self.spin[i][j][self.size-1]
            else:
                front_spin = self.spin[i][j][k-1]
                    
            if k==(self.size-1):
                back_spin = self.spin[i][j][0]
            else:
                back_spin = self.spin[i][j][k+1]
                
            return -self.spin[i][j][k] * self.K * (left_spin + right_spin + top_spin + bottom_spin + front_spin + back_spin)

            




    def magnetisation(self):
        sum = 0
        
        if self.type=='1d':
            for i in range(self.size):
                sum += self.spin[i]
            return sum/(self.size)
        
        elif self.type=='2d':
            for i in range(self.size):
                for j in range(self.size):
                    sum += self.spin[i][j]
            return sum/(self.size **2)
            
        elif self.type=='3d':
            for i in range(self.size):
                for j in range(self.size):
                    for k in range(self.size):
                        sum += self.spin[i][j][k]
            return sum/(self.size **3)
    '''returns average magnetisation per element'''
    
    

    def delta_E(self,i,j,k=0):
        
        if self.type == '1d':
    
            if i == 0:
                left_spin = self.spin[self.size-1]
            else:
                left_spin = self.spin[i-1]

            if i == (self.size-1):
                right_spin = self.spin[0]
            else:
                right_spin = self.spin[i+1]

            energy_change = 2*self.spin[i] * self.K * (left_spin + right_spin)
            
            return energy_change

        
        if self.type == '2d':
    
            if i == 0:
                left_spin = self.spin[self.size-1][j]
            else:
                left_spin = self.spin[i-1][j]

            if i == (self.size-1):
                right_spin = self.spin[0][j]
            else:
                right_spin = self.spin[i+1][j]

            if j == 0:
                top_spin = self.spin[i][self.size-1]
            else:
                top_spin = self.spin[i][j-1]

            if j == (self.size-1):
                bottom_spin = self.spin[i][0]
            else:
                bottom_spin = self.spin[i][j+1]

            energy_change = 2*self.spin[i][j] * self.K * (left_spin + right_spin + top_spin + bottom_spin)
        
            return energy_change
            
            
        if self.type=='3d':

            if i == 0:
                left_spin = self.spin[self.size-1][j][k]
            else:
                left_spin = self.spin[i-1][j][k]
                    
            if i == (self.size-1):
                right_spin = self.spin[0][j][k]
            else:
                right_spin = self.spin[i+1][j][k]
        
            if j == 0:
                top_spin = self.spin[i][self.size-1][k]
            else:
                top_spin = self.spin[i][j-1][k]
                
            if j == (self.size-1):
                bottom_spin = self.spin[i][0][k]
            else:
                bottom_spin = self.spin[i][j+1][k]
            
            if k==0:
                front_spin = self.spin[i][j][self.size-1]
            else:
                front_spin = self.spin[i][j][k-1]
                
            if k==(self.size-1):
                back_spin = self.spin[i][j][0]
            else:
                back_spin = self.spin[i][j][k+1]

            energy_change = 2*self.spin[i][j][k] * self.K * (left_spin + right_spin + top_spin + bottom_spin + front_spin + back_spin) 
            
            return energy_change



    def metropolis(self,*args):
    
        if self.type=='1d':
            for n in range(self.size):

                i = int(random.randint(0,self.size -1))
                Delta = self.delta_E(i,0)
                prob = random.random()
                if Delta<=0:
                    self.spin[i] = - self.spin[i]
                if Delta>0:
                    if prob < np.exp(-self.beta*Delta):
                        self.spin[i] = - self.spin[i]
              
        if self.type=='2d':   
            for n in range(self.size**2):

                i = int(random.randint(0,self.size -1))
                j = int(random.randint(0,self.size -1))
                Delta = self.delta_E(i,j)
                prob = random.random()
                if Delta<=0:
                    self.spin[i][j] = - self.spin[i][j]
                if Delta>0:
                    if prob < np.exp(-self.beta*Delta):
                        self.spin[i][j] = - self.spin[i][j]
                                    
        if self.type=='3d':
            for n in range(self.size**3):

                i = int(random.randint(0,self.size -1))
                j = int(random.randint(0,self.size -1))
                k = int(random.randint(0,self.size -1))
                Delta = self.delta_E(i,j,k)
                prob = random.random()
                if Delta<=0:
                    self.spin[i][j][k] = - self.spin[i][j][k]
                if Delta>0:
                    if prob < np.exp(-self.beta*Delta):
                        self.spin[i][j][k] = - self.spin[i][j][k]
                   




    def update(self,*args):
        
        if self.type=='1d':  
            for n in range(self.size):
                i = int(random.randint(0,self.size -1))
                Delta = self.delta_E(i,0)
                prob = random.random()
                if Delta<=0:
                    self.spin[i] = - self.spin[i]
                if Delta>0:
                    if prob < np.exp(-self.beta*Delta):
                        self.spin[i] = - self.spin[i]


        else:
            for n in range(self.size**2):
                i = int(random.randint(0,self.size -1))
                j = int(random.randint(0,self.size -1))
                Delta = self.delta_E(i,j)
                #print Delta
                prob = random.random()
                if Delta<=0:
                    self.spin[i][j] = - self.spin[i][j]
                if Delta>0:
                    if prob < np.exp(-self.beta*Delta):
                        self.spin[i][j] = - self.spin[i][j]

        self.flat_spin = np.reshape(self.spin,(1,np.product(self.spin.shape)))
        return self.Colour_UpdateRule(0)



    def animate(self, speed, frames,save=False):
        self.fig = plt.figure(figsize= (8,8))
        self.anim = FuncAnimation(self.fig, self.update,frames=frames)
        if save==True:
            self.anim.save('lattice_animation.gif', writer=PillowWriter(fps=24))
        plt.show()
        
    def change_h(self,h):
        self.h = h

    def change_temp(self,T):
        self.temp = T
        self.beta = (T)**(-1.0)

    def Colour_UpdateRule(self,i):
        if self.type=='1d':
            clourArray = np.where(self.flat_spin[ i ]  == 1, '#483d8b','#ffd700')
            return plt.scatter(self.points[:,0],np.zeros(self.size),c = clourArray.T.flatten())
    
        if self.type=='2d':
            clourArray = np.where(self.flat_spin[ i ]  == 1, '#483d8b','#ffd700')
            return plt.scatter(self.points[:,0],self.points[:,1],c = clourArray.T.flatten())
            
    def Display(self,factor=1):
        i=0
        self.Renormalise(factor)
        self.flat_spin = np.reshape(self.renorm_spin,(1,np.product(self.renorm_spin.shape)))
        if self.type=='1d':
            clourArray = np.where(self.flat_spin[ i ]  == 1, '#483d8b','#ffd700')
            plt.scatter(self.renorm_points[:,0],np.zeros(self.renorm_size),c = clourArray.T.flatten())
    
        if self.type=='2d':
            clourArray = np.where(self.flat_spin[ i ]  == 1, '#000000','#FF0000')
            plt.scatter(self.renorm_points[:,0],self.renorm_points[:,1],c = clourArray.T.flatten())
        plt.show()
            
    '''sets renormalised lattice to be accurate for current spin'''
    def Renormalise(self,factor):
        new_size = int(self.renorm_size/factor)
        if self.type=='1d':
            self.renorm_points = np.array([[a] for a in np.linspace(0,new_size-1,new_size)])
            new_spins = np.zeros((new_size))
            for i in range(new_size):
                new_spins[i] = self.renorm_spin[int(factor*i)]
            self.renorm_spin = new_spins
            if factor==2:
                self.renorm_K = 0.5 * np.log(np.cosh(2*self.K))

        if self.type=='2d':
            new_spins = np.zeros((new_size,new_size))
            for i in range(new_size):
                for j in range(new_size):
                    new_spins[i][j] = self.renorm_spin[int(factor*i)][int(factor*j)]
                    
            self.renorm_points = np.array([[a, b] for a in np.linspace(0,new_size-1,new_size) for b in np.linspace(0,new_size-1,new_size)])
            self.renorm_spin = new_spins
            
        self.renorm_size = new_size
        

    
'''WRONG COUPLING UPDATE'''
def block_renormalise(lattice,factor):
    old_spins = lattice.get_spins()
    old_coupling = lattice.get_coupling()
    type = lattice.get_type()
    old_size = lattics.get_size()
    new_size = int(old_size/factor)
    new_coupling = old_coupling**2
    if type=='1d':
        new_spins = np.zeros((new_size))
        for i in range(new_size):
            new_spins[i] = old_spins[int(factor*i)]
    
    


def Initialise_Random_State(N,T,K,type='1d'):
    Rand_state = state(N,K,T,type=type)
    if type=='1d':
        for i in range(Rand_state.size):
            chance = random.randint(0,1)
            if chance > 0.5:
                Rand_state.spin[i] = 1
                '''dipole spin up'''
            else:
                Rand_state.spin[i] = -1
                '''dipole spin down'''

    if type=='2d':
        for i in range(Rand_state.size):
            for j in range(Rand_state.size):
                if random.randint(0,1) > 0.5:
                    Rand_state.spin[i][j] = 1
                    '''dipole spin up'''
                else :
                    Rand_state.spin[i][j] = -1
                '''dipole spin down'''
                
    if type=='3d':
        for i in range(Rand_state.size):
            for j in range(Rand_state.size):
                for k in range(Rand_state.size):
                    if random.randint(0,1) > 0.5:
                        Rand_state.spin[i][j][k] = 1
                        '''dipole spin up'''
                    else :
                        Rand_state.spin[i][j][k] = -1
                        '''dipole spin down'''

    
    return Rand_state

'''Initialising a random state to start the routine'''




