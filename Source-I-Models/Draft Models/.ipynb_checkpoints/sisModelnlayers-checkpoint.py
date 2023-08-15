import numpy as np
import math
from matplotlib import pyplot as plt
import random
from operator import add

class SiSModel:
    
    v_lsr = 10 #km/s
    
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
        vy = vyprime * np.cos(self.theta) - vzprime * np.sin(self.theta)
        vz = vyprime * np.sin(self.theta) + vzprime * np.cos(self.theta)
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
    
    
    
    ### Any variable is contained within the mask
    def constrain_var_radially(self, r, phi, zprime, var):
        mask = (r > self.r_inner(zprime)) & (r < self.r_outer(zprime))
        var_lim = np.where(mask, var, np.nan)
        return var_lim
    
                      
    
    ### r, phi, z, vr, vphi and vz are constrained in the radial mask
    def constrain_radially(self, r, phi, zprime, vr, vphi, vz):
        mask = (r > self.r_inner(zprime)) & (r < self.r_outer(zprime))
        
        r_lim = np.where(mask, r, np.nan)
        phi_lim = np.where(mask, phi, np.nan)
        z_lim = np.where(mask, zprime, np.nan)
        v_r_lim = np.where(mask, vr, np.nan)
        v_phi_lim = np.where(mask, vphi, np.nan)
        v_z_lim = np.where(mask, vz, np.nan)
        
        return r_lim, phi_lim, z_lim, v_r_lim, v_phi_lim, v_z_lim
    
    
    
    
    
    def signal_strength_outflow_coordinates(self, r, phi, z_prime, exp_factor, los):
        if los == "x" or los == "X":
            return self.x_los_signal_strength(r, phi, z_prime, exp_factor)
        elif los == "y" or los == "Y":
            return self.y_los_signal_strength(r, phi, z_prime, exp_factor)
        else:
            return false
        
        
    def signal_strength_sky_coordinates(self, x, y, z, exp_factor, los):
        r, phi, z_prime = self.sky_to_outflow(x, y, z)
        
        if los == "x" or los == "X":
            return self.x_los_signal_strength(r, phi, z_prime, exp_factor)
        elif los == "y" or los == "Y":
            return self.y_los_signal_strength(r, phi, z_prime, exp_factor)
        else:
            return false
    
    
    def x_los_signal_strength(self, r, phi, z_prime, exp_factor):           
        
        x_model = r * np.cos(phi)
        y_model = r * np.sin(phi)
        z_model = z_prime
        
        r_inner = self.r_inner(z_prime)
        r_outer = self.r_outer(z_prime)
        
        x_sky = x_model
        y_sky = y_model * np.cos(self.theta) - z_model * np.sin(self.theta)
        z_sky = y_model * np.sin(self.theta) + z_model * np.cos(self.theta)
        
        signal_strength = np.zeros_like(y_model)
        
        x_model = self.constrain_var_radially(r, phi, z_prime, x_model)
        y_model = self.constrain_var_radially(r, phi, z_prime, y_model)
        
        x_l, y_l = y_model.shape
        
        for x in range(x_l):
            for y in range(y_l):
                if np.isnan(y_model[x, y]):
                    signal_strength[x, y] = np.nan
                else:
                    outer_cord_length = np.sqrt(np.abs(np.power(r_outer[x, y], 2) - np.power(y_model[x, y], 2)))

                    if np.abs(y_model[x, y]) < r_inner[x, y]:
                        inner_cord_length = np.sqrt(np.abs(np.power(r_inner[x, y], 2) - np.power(y_model[x, y], 2)))

                        if x_model[x, y] > 0:
                            strength = -(outer_cord_length - x_model[x, y])
                            signal_strength[x, y] = np.exp(strength * exp_factor)
                        else:
                            strength = -(outer_cord_length - x_model[x, y] - inner_cord_length * 2)
                            signal_strength[x, y] = np.exp(strength * exp_factor)

                    else:
                        strength = -(outer_cord_length - x_model[x, y])
                        signal_strength[x, y] = np.exp(strength * exp_factor)
        
        return signal_strength 
    
    
    def y_los_signal_strength(self, r, phi, z_prime, exp_factor):  
        ### DOES NOT TAKE SKY ANGLE INTO ACCOUNT ###
        
        x_model = r * np.cos(phi)
        y_model = r * np.sin(phi)
        z_model = z_prime
        
        r_inner = self.r_inner(z_prime)
        r_outer = self.r_outer(z_prime)
        
        x_sky = x_model
        y_sky = y_model * np.cos(self.theta) - z_model * np.sin(self.theta)
        z_sky = y_model * np.sin(self.theta) + z_model * np.cos(self.theta)
        
        signal_strength = np.zeros_like(x_model)
        
        x_model = self.constrain_var_radially(r, phi, z_prime, x_model)
        y_model = self.constrain_var_radially(r, phi, z_prime, y_model)
        
        x_l, y_l = y_model.shape
        
        for y in range(y_l):
            for x in range(x_l):
                if np.isnan(x_model[x, y]):
                    signal_strength[x, y] = np.nan
                else:
                    outer_cord_length = np.sqrt(np.abs(np.power(r_outer[x, y], 2) - np.power(x_model[x, y], 2)))

                    if np.abs(x_model[x, y]) < r_inner[x, y]:
                        inner_cord_length = np.sqrt(np.abs(np.power(r_inner[x, y], 2) - np.power(x_model[x, y], 2)))

                        if y_model[x, y] > 0:
                            strength = -((outer_cord_length) - y_model[x, y])
                            signal_strength[x, y] = np.exp(strength * exp_factor)
                        else:
                            strength = -((outer_cord_length) - y_model[x, y] - inner_cord_length * 2)
                            signal_strength[x, y] = np.exp(strength * exp_factor)
                        
                    else:
                        strength = -((outer_cord_length) - y_model[x, y])
                        signal_strength[x, y] = np.exp(strength * exp_factor)
        
        return signal_strength
        
        

    # Outflow velocity change rates #
    def v_radial_dropoff(self, z):
        return 1.


    def v_phi_dropoff(self, z):
        return 1. - np.abs(z)/self.z_no_rotation
    
    def v_z_dropoff(self, z):
        return 1.
    
                      
                      
    
    

    ### Velocity Field Functions ###
    # Outflow (cylindrical) coordinates in [r, phi, z] for every point in space created in model #
    
    
    def velocity_model_field(self, x, y, z):
        r, phi, zprime = self.sky_to_outflow(x, y, z)        
    
        mask = (r > self.r_inner(zprime)) & (r < self.r_outer(zprime))
        vr = np.where(mask, self.v_radial * self.v_radial_dropoff(z), np.nan)

        # Azimuthal rotation of vphi0 next to disk, drops linearly to 0 at height z_no_rotation #
        vphi = np.where(z < self.z_no_rotation, self.v_phi0 * self.v_phi_dropoff(z), 0.)
        vphi = np.where(mask, vphi, np.nan)
        
        # Flows away from disk at constant velocity vz0
        vz = np.where(z > 0., self.vz_0 * self.v_z_dropoff(z), -self.vz_0 * self.v_z_dropoff(z))
        vz = np.where(mask, vz, np.nan)
        
                      
        vx, vy, vz = self.outflow_to_sky_velocity(phi, vr, vphi, vz)
        
        vx = self.add_random_velocity_component(vx)
        vy = self.add_random_velocity_component(vy)
        vz = self.add_random_velocity_component(vz)
        
        
        return vx, vy, vz

    
    
    
    
    def velocity_model_field_turbulent(self, x, y, z) :
        r, phi, zprime = self.sky_to_outflow(x, y, z)        
    
        mask = (r > self.r_inner(zprime)) & (r < self.r_outer(zprime))
        vr = np.where(mask, self.v_turbulent * self.v_radial_dropoff(z), np.nan)

        # Azimuthal rotation of vphi0 next to disk, drops linearly to 0 at height z_no_rotation #
        vphi = np.where(z < self.z_no_rotation, self.v_phi0 * self.v_phi_dropoff(z), 0.)
        vphi = np.where(mask, vphi, np.nan)
        
        # Flows away from disk at constant velocity vz0
        vz = np.where(z > 0., self.vz_0 * self.v_z_dropoff(z), -self.vz_0 * self.v_z_dropoff(z))
        vz = np.where(mask, vz, np.nan)
        
                      
        vx, vy, vz = self.outflow_to_sky_velocity(phi, vr, vphi, vz)
        
        vx = self.add_random_velocity_component(vx)
        vy = self.add_random_velocity_component(vy)
        vz = self.add_random_velocity_component(vz)
        
        
        return vx, vy, vz
    
    
    def velocity_model_field_inward(self, x, y, z) :
        r, phi, zprime = self.sky_to_outflow(x, y, z)        
    
        mask = (r > self.r_inner(zprime)) & (r < self.r_outer(zprime))
        vr = np.where(mask, self.v_inward * self.v_radial_dropoff(z), np.nan)

        # Azimuthal rotation of vphi0 next to disk, drops linearly to 0 at height z_no_rotation #
        vphi = np.where(z < self.z_no_rotation, self.v_phi0 * self.v_phi_dropoff(z), 0.)
        vphi = np.where(mask, vphi, np.nan)
        
        # Flows away from disk at constant velocity vz0
        vz = np.where(z > 0., self.vz_0 * self.v_z_dropoff(z), -self.vz_0 * self.v_z_dropoff(z))
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
    def position_velocity(self, axis, exp_factor):
        def pv(ax, z, vmin, vmax, nvbins, axis, exp_factor):
            x, y = np.meshgrid( np.linspace(-400,400,501), 
                               np.linspace(-400,400,501) )
                # using 240 AU x 240 AU slices for this example
                # possible addition to the code is to check that outflow fits within the slice
                      
            
            vx, vy, vz = self.velocity_model_field(x, y, z)
            vxt, vyt, vzt = self.velocity_model_field_turbulent(x, y, z)
            vxi, vyi, vzi = self.velocity_model_field_inward(x, y, z)

                # 2D [y,x] array of velocity vectors in plane z
                # we only care about vy, the velocity along the line of sight
                
            
            r, phi, zprime = self.sky_to_outflow(x, y, z)
                
            if axis == "x":
                opacity = self.y_los_signal_strength(r, phi, zprime, exp_factor)
                v_y_adj = -1 * vy + SiSModel.v_lsr
                v_y_t_adj = -1 * vyt + SiSModel.v_lsr               
                v_y_i_adj = -1 * vyi + SiSModel.v_lsr

                # add the source VLSR
                vlos=[]
                vlosi=[]
                vlost=[]
                for i in range(0,len(x)) :
                    vl,binedges = np.histogram(v_y_adj[:, i], range=(vmin, vmax), bins=nvbins, weights=opacity[:, i])
                    vlt,binedges = np.histogram(v_y_t_adj[:, i], range=(vmin, vmax), bins=nvbins, weights=opacity[:, i])
                    vli,binedges = np.histogram(v_y_i_adj[:, i], range=(vmin, vmax), bins=nvbins, weights=opacity[:, i])

                    vlos.append(vl)
                    vlost.append(vlt)
                    vlosi.append(vli)
                
                v_final = list(map(add, vlos, vlost))
                v_final = list(map(add, v_final, vlosi))
                
                ax.imshow( v_final, origin='lower', aspect='auto', extent=(vmin, vmax, x[0,0], x[-1,-1]) )
                ax.annotate("z = %.0f AU" % z, (.05, .9), xycoords="axes fraction", color='white')
                
            elif axis == "y":
                opacity = self.x_los_signal_strength(r, phi, zprime, exp_factor)
                v_x_adj = -1 * vx + SiSModel.v_lsr
                v_x_t_adj = -1 * vxt + SiSModel.v_lsr               
                v_x_i_adj = -1 * vxi + SiSModel.v_lsr

                # add the source VLSR
                vlos=[]
                vlosi=[]
                vlost=[]
                for i in range(0,len(y)) :
                    vl,binedges = np.histogram(v_x_adj[i, :], range=(vmin, vmax), bins=nvbins, weights=opacity[i, :])                   
                    vlt,binedges = np.histogram(v_x_t_adj[i, :], range=(vmin, vmax), bins=nvbins, weights=opacity[i, :])
                    vli,binedges = np.histogram(v_x_i_adj[i, :], range=(vmin, vmax), bins=nvbins, weights=opacity[i, :])

                    vlos.append(vl)
                    vlost.append(vlt)
                    vlosi.append(vli)
                
                v_final = list(map(add, vlos, vlost))
                v_final = list(map(add, v_final, vlosi))
                
                ax.imshow(v_final, origin='lower', aspect='auto', extent=(vmin, vmax, y[0,0], y[-1,-1]) )
                ax.annotate("z = %.0f AU" % z, (.05, .9), xycoords="axes fraction", color='white')
            else:
                return false;


        # make a grid of 9 position-velocity plots at intervals z_step along outflow axis
        nx = 3
        ny = 3
        fig, ax = plt.subplots(nx, ny, sharex=True, sharey=True, figsize=(10, 10))

        z = 0.
        z_step = 50.
            # z starting value and step size in AU

        vmin = SiSModel.v_lsr - 20.
        vmax = SiSModel.v_lsr + 20.
        nvbins = 50
            # set velocity range and resolution

        for i in range(0, nx) :
            for j in range(0, ny) :
                pv(ax[i, j], z, vmin, vmax, nvbins, axis, exp_factor)
                if i == nx - 1 :
                    ax[i, j].set_xlabel("VLSR (km/sec)")
                if j == 0 :
                    ax[i, j].set_ylabel(axis + " offset (AU)")
                z += z_step

        plt.tight_layout()
        plt.show()
        
        