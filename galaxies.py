import numpy as np
import matplotlib.pyplot as plt
import numpy.random as random
from scipy.integrate import quad
from scipy.interpolate import interp1d

from myTreecode import Star


time_conversion_factor = 14.910128      # 1 IU in Myrs
energy_conversion_factor = 1/1524.834   # sun kinetic energy in internal units
velocity_conversion_factor = 1 / 15.248335  # 1 IU in Km/s 
time_conversion_factor_s = 470563626066861.375000 # 1 IU in s


class Plummer:
    
    def __init__(self,N,b,m=1):
        self.N = int(N) 
        self.b = b 
        self.m = m 
        self.M = N*self.m 
        self.rho0 = 3*self.M/(4*np.pi*self.b**3)
 
        print(f"Plummer sphere of {self.N} stars and total mass {self.M:.2f} solar masses. "
              f"Scale parameter b is {self.b:.2f} parsecs.")

                
        self.v_typical = np.sqrt(self.M/self.b)  # from virial theorem
        self.t_cross = self.b / self.v_typical        
        self.t_relax = (0.1 * self.N / np.log(self.N)) * self.t_cross

        print(f"Characteristic velocity is {self.v_typical:.2e} in internal units, "
              f"{self.v_typical * velocity_conversion_factor:.2e} km/s. ")
        print(f"Crossing time is {self.t_cross:.2e} in internal units, "
              f"{self.t_cross * time_conversion_factor:.2e} Myr. "
              f"Relaxation time is {self.t_relax:.2e} in internal units, "
              f"{self.t_relax * time_conversion_factor:.2e} Myr.")

        # initialize system
        
        self.t = [0]
        self.system = []

        initial_conditions = []
        
        for i in range(self.N):
            
            u = random.uniform(0,1)
            v = random.uniform(0,1)
            w = random.uniform(0,1)
            r = self.b * np.sinh(np.arctanh(u**(1/3))) 
            theta = np.arccos(1-2*v) 
            phi = 2*np.pi*w
            
            x = np.array([r*np.sin(theta)*np.cos(phi),r*np.sin(theta)*np.sin(phi),r*np.cos(theta)])
            Psi_x = self.M/np.sqrt(r**2 + self.b**2)
            
            # extract a point from velocity distribution ranging from 0 to q = 1 (v = v_esc)
            norm = 7*np.pi/512   ### serve?
            while True:
                q = random.uniform(0,1) 
                y = random.uniform(0,0.093/norm) 
                if y < (1/norm) * q**2 * (1-q**2)**(7/2):
                    break
                    
            v_magn = q * np.sqrt(2*Psi_x)
            theta = np.arccos(1-2*random.uniform(0,1)) 
            phi = 2*np.pi*random.uniform(0,1)
            
            v = np.array([v_magn*np.sin(theta)*np.cos(phi),v_magn*np.sin(theta)*np.sin(phi),v_magn*np.cos(theta)])
            
            initial_conditions.append(Star(self.m, x, v))
            
        self.system.append(np.array(initial_conditions))



class Hernquist:
    
    def __init__(self,N,a,m=1,M_bh=0,plot=False):
        self.N = int(N) 
        self.a = a 
        self.m = m 
        self.M = N*self.m   # stellar mass
        self.rho0 = self.M / (2*np.pi*self.a**3) 
        self.M_bh = M_bh

        
        print(f"Hernquist model of {self.N} stars and total mass {self.M:.2f} solar masses. "
              f"Scale parameter a is {self.a:.2f} parsecs.")
        if M_bh > 0: print(f"Central black hole of mass {M_bh:.2f}")

        self.v_typical = np.sqrt((self.M + M_bh)/ self.a)  # from virial theorem
        self.t_cross = self.a / self.v_typical        
        self.t_relax = (0.1 * self.N / np.log(self.N)) * self.t_cross

        print(f"Characteristic velocity is {self.v_typical:.2e} in internal units, "
              f"{self.v_typical * velocity_conversion_factor:.2e} km/s. ")
        print(f"Crossing time is {self.t_cross:.2e} in internal units, "
              f"{self.t_cross * time_conversion_factor:.2e} Myr. "
              f"Relaxation time is {self.t_relax:.2e} in internal units, "
              f"{self.t_relax * time_conversion_factor:.2e} Myr.")


        ### initialize arrays for numeric DF calculation ###
  
        r_array = np.logspace(-4,4,100000) * self.a  
        nu = self.rho0 / self.M / ((r_array/a) * (1+r_array/a)**3)      # mass probability distribution
        Psi = self.M / self.a / (1+r_array/a)  +  self.M_bh/r_array     # potential
        nu = nu[::-1] 
        Psi = Psi[::-1]
        dnu_dPsi = np.gradient(nu, Psi)
        
        E_array = np.logspace(np.log10(min(Psi)), np.log10(max(Psi)),1000) 
        
        Y = []
        for E in E_array: 
            Y.append(quad(lambda psi: np.interp(psi, Psi, dnu_dPsi) / np.sqrt(E - psi), 0, E)[0])

        DF_num = 1/(2**(3/2) * np.pi**2) * np.gradient(Y,E_array)
        DF_fun = interp1d(E_array, DF_num, kind='cubic', fill_value="extrapolate")

        
        # initialize system

        
        self.t = [0]
        self.system = []

        initial_conditions = []

        if M_bh > 0:
            initial_conditions.append(Star(M_bh,np.array([0,0,0]),np.array([0,0,0])))
            self.N = self.N + 1

        for i in range(N):
            
            #inizialize position 
            
            m_r = random.uniform(0,self.M)
            x = 2 * np.pi * self.rho0 * self. a / m_r
            y = 1 / self.a
            r = (-y-np.sqrt(x))/(y**2-x) 
            theta = np.arccos(1-2*random.uniform(0,1)) 
            phi = 2*np.pi*random.uniform(0,1)
            x = np.array([r*np.sin(theta)*np.cos(phi),r*np.sin(theta)*np.sin(phi),r*np.cos(theta)])

            # initialize velocity

            # inverse sampling
            Psi_particle = self.M/(self.a + r) + self.M_bh / r
            E_array = np.linspace(0,Psi_particle,100)

            F = [0] 
            for k in range(1,len(E_array)):
                F.append(F[k-1] + quad(lambda e: 4*np.pi * DF_fun(e) * np.sqrt(2 * (Psi_particle - e)), E_array[k-1],E_array[k])[0])

            inverse_integral = interp1d(F,E_array,kind='linear')

            nu_r = self.rho0 / (r/self.a*(1+r/self.a)**3) / self.M  # normalization
            F_e = random.uniform(0,nu_r)
            if F_e < max(F):
                E = inverse_integral(F_e) 
            else:
                E = inverse_integral(max(F)) 
 
            v_magn = np.sqrt(2*(Psi_particle - E))
            theta = np.arccos(1-2*random.uniform(0,1)) 
            phi = 2*np.pi*random.uniform(0,1)

            v = np.array([v_magn*np.sin(theta)*np.cos(phi),v_magn*np.sin(theta)*np.sin(phi),v_magn*np.cos(theta)])

            initial_conditions.append(Star(self.m, x, v))
        
        self.system.append(np.array(initial_conditions))







