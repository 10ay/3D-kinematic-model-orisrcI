import numpy as np
import math
from matplotlib import pyplot as plt
import random
from operator import add
from astropy.io import fits

class SiSModel:
    
    v_lsr = 5 #km/s
    
    def __init__(self, r_in, r_out, r_in_lin, r_out_lin, r_in_par, r_out_par, r_cut, theta, v_r_out, v_r_in, v_phi, v_z, v_rand, z_rot_limit, seed):

        # Primary outflow shape parameters
        # ------------------------------------------------------------------------------------#
        self.r0_inner = r_in           #in AU
        self.r0_outer = r_out         #in AU
        self.r1_inner = r_in_lin       #Dimensionless Rate
        self.r1_outer = r_out_lin     #Dimensionless Rate
        self.r2_inner = r_in_par       #in AU^0.5
        self.r2_outer = r_out_par     #in AU^0.5
        self.r_cut_percentage = r_cut
        self.theta = theta     #in rad
        random.seed(seed)
        # ------------------------------------------------------------------------------------#
        
        
        # Velocity parameters
        # ------------------------------------------------------------------------------------#
        self.v_radial = v_r_out      #in km/s
        self.v_inward = v_r_in
        self.vz_0 = v_z      #in km/s
        self.v_phi0 = v_phi  #in km/s
        self.v_turbulent = v_rand   #in km/s
        # ------------------------------------------------------------------------------------#
        
        
        # Limitation parameters
        # ------------------------------------------------------------------------------------#
        self.z_no_rotation = z_rot_limit #in AU
        # ------------------------------------------------------------------------------------#
        
        

        
    ### Dimension Functions ###
    def r_inner(self, z):
        return self.r0_inner + (np.abs(z) * self.r1_inner) + (np.sqrt(np.abs(z)) * self.r2_inner)
    
    
    def r_outer(self, z):
        return self.r0_outer + (np.abs(z) * self.r1_outer) + (np.sqrt(np.abs(z)) * self.r2_outer)
    
    
    
    def r_cut(self, z):
        return (self.r_outer(z) - self.r_inner(z)) * self.r_cut_percentage + self.r_inner(z)
    
    
   
    
    ###Co-ordinate Converters 
    
    ### # convert from (x, y, z) sky coordinates to (x', y', z') coordinates aligned with outflow
    def sky_to_prime(self, x_sky, y_sky, z_sky):
        x_prime = x_sky
        y_prime = y_sky * np.cos(self.theta) - z_sky * np.sin(self.theta)
        z_prime = y_sky * np.sin(self.theta) + z_sky * np.cos(self.theta)
        return x_prime, y_prime, z_prime
        
    def prime_to_sky(self, x_prime, y_prime, z_prime):
        x = x_prime
        y = y_prime * np.cos(self.theta) - z_prime * np.sin(self.theta)
        z = y_prime * np.sin(self.theta) + z_prime * np.cos(self.theta)
        return x, y, z
        

        
        
    ### # convert from (x', y', z') to (r, y, z') outflow frame coordinates (cylindrical).
    # x-axis is line-of-sight axis, y-axis is RA axis, z-axis is DEC axis #
    def prime_to_outflow(self, x_prime, y_prime, z_prime):    
        r = np.sqrt(np.power(x_prime, 2) + np.power(y_prime, 2))
        phi = np.arctan2(y_prime, x_prime)
        return r, phi, z_prime
    
    
    
    def outflow_to_prime(self, r, phi, z_prime):
        x_prime = r*np.cos(phi)
        y_prime = r*np.sin(phi)
        z_prime = z_prime
        return x_prime, y_prime, z_prime
    

    
    def sky_to_outflow(self, x, y, z):
        x_prime, y_prime, z_prime = self.sky_to_prime(x, y, z)
        r, phi, z_prime = self.prime_to_outflow(x_prime, y_prime, z_prime)
        return r, phi, z_prime
    
    
    
    def outflow_to_sky(self, r, phi, z_prime):
        x_prime, y_prime, z_prime = self.outflow_to_prime(r, phi, z_prime)
        x, y, z = self.prime_to_sky(x_prime, y_prime, z_prime)
        return x, y, z
    

    
    ### Velocity Converters
    def outflow_to_prime_velocity(self, phi, vr, vphi, vzprime):
        vxprime =  vr * np.cos(phi) - vphi * np.sin(phi)
        vyprime = vr * np.sin(phi) + vphi * np.cos(phi)
        vzprime = vzprime
        return vxprime, vyprime, vzprime
    
    
    def prime_to_sky_velocity(self, vxprime, vyprime, vzprime):
        vx = vxprime
        vy = vyprime * np.cos(self.theta) + vzprime * np.sin(self.theta)
        vz = -vyprime * np.sin(self.theta) + vzprime * np.cos(self.theta)
        return vx, vy, vz
    
    def outflow_to_sky_velocity(self, phi, vr, vphi, vzprime):
        vxprime, vyprime, vzprime = self.outflow_to_prime_velocity(phi, vr, vphi, vzprime)
        vx, vy, vz = self.prime_to_sky_velocity(vxprime, vyprime, vzprime)
        return vx, vy, vz
    
    
    
    ### Dealing with random velocity components
    def add_random_velocity_component(self, v):
        v_mutate = []
        for vel in np.nditer(v):
            v_mutate.append(vel+random.uniform(-self.v_turbulent, self.v_turbulent))
        v_mutate = np.reshape(v_mutate, v.shape)
        return v_mutate
    
        

    # Outflow velocity change rates #
    def v_radial_dropoff(self, z):
        return 1.


    def v_phi_dropoff(self, z):
        return 1. - np.abs(z)/self.z_no_rotation
    
    def v_z_dropoff(self, z):
        return 1.
    
                      
                      
    
    

    ### Velocity Field Functions ###
    # Outflow (cylindrical) coordinates in [r, phi, z] for every point in space created in model #
    
    
    def velocity_model_field(self, r, phi, zprime):
    
        mask = (r > self.r_inner(zprime)) & (r < self.r_outer(zprime))
        vr = np.where(mask, self.v_radial * self.v_radial_dropoff(zprime) , np.nan)

        # Azimuthal rotation of vphi0 next to disk, drops linearly to 0 at height z_no_rotation #
        vphi = np.where(np.abs(zprime) < self.z_no_rotation, self.v_phi0 * self.v_phi_dropoff(zprime), np.nan)
        vphi = np.where(mask, vphi, 0.)
        
        # Flows away from disk at constant velocity vz0
        vz = np.where(zprime > 0., self.vz_0 * self.v_z_dropoff(zprime), -self.vz_0 * self.v_z_dropoff(zprime))
        vz = np.where(mask, vz, np.nan)
        
                      
        vx, vy, vz = self.outflow_to_sky_velocity(phi, vr, vphi, vz)
        
        vx = self.add_random_velocity_component(vx)
        vy = self.add_random_velocity_component(vy)
        vz = self.add_random_velocity_component(vz)
        
        
        return vx, vy, vz

    
    
    
    
    
    
    ###Density functions
    def density_prime(self, xprime, yprime, zprime) :
        r, phi, z = self.prime_to_outflow(xprime, yprime, zprime)
        mask = (r > self.r_inner(z)) & (r < self.r_outer(z))
        density = np.where(mask, 1., np.nan)
        return density
    
    def density_sky(self, x, y, z) :
        r, phi, z = self.sky_to_outflow(x, y, z)
        mask = (r > self.r_inner(z)) & (r < self.r_outer(z))
        density = np.where(mask, 1., np.nan)
        return density

    
    
    
    
                      
                      
    ###PV Plot
    def position_velocity(self, exp_factor):
        def pv(ax, z, vmin, vmax, nvbins, exp_factor):
            
            #z= np.abs(z)
            
            
            x, y = np.meshgrid( np.linspace(-400,400,501), 
                               np.linspace(-400,400,501) )
                # using 240 AU x 240 AU slices for this example
                # possible addition to the code is to check that outflow fits within the slice
                      
            r, phi, zprime = self.sky_to_outflow(x, y, z)
            
            
            #zprime = np.abs(zprime)
            
            vx, vy, vz = self.velocity_model_field(r, phi, zprime)
                # 2D [y,x] array of velocity vectors in plane z
                # we only care about vy, the velocity along the line of sight
                
            
            if np.any(zprime<0):
                v_y_adj = -vy + SiSModel.v_lsr
            else:
                v_y_adj = vy + SiSModel.v_lsr

                # add the source VLSR
            vlos=[]
            for i in range(0,len(x)) :
                vl,binedges = np.histogram(v_y_adj[:, i], range=(vmin, vmax), bins=nvbins)
                vlos.append(vl)                
                
            ax.imshow( vlos, origin='lower', aspect='auto', extent=(vmin, vmax, x[0,0], x[-1,-1]) )
            ax.annotate("z = %.0f AU" % z, (.05, .9), xycoords="axes fraction", color='white')
    

        # make a grid of 12  position-velocity plots at intervals z_step along outflow axis
        nx = 3
        ny = 4
        fig, ax = plt.subplots(nx, ny, sharex=True, sharey=True, figsize=(10, 10))

        z = -400.
        z_step = 80.
            # z starting value and step size in AU

        vmin = SiSModel.v_lsr - 20.
        vmax = SiSModel.v_lsr + 20.
        nvbins = 50
            # set velocity range and resolution

        for i in range(0, nx) :
            for j in range(0, ny) :
                pv(ax[i, j], z, vmin, vmax, nvbins, exp_factor)
                if i == nx - 1 :
                    ax[i, j].set_xlabel("VLSR (km/sec)")
                if j == 0 :
                    ax[i, j].set_ylabel("x offset (AU)")
                z += z_step

        plt.tight_layout()
        plt.show()
        