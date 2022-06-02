import numpy as np
import astropy.io.fits
import matplotlib.pyplot as plt
import scipy as sym
import scipy.integrate as integrate
from scipy import log,exp,sqrt,stats
from astropy import units as u
import astropy.constants as const
from scipy.optimize import curve_fit
from astropy.stats import biweight_location, biweight_scale, bootstrap
from astropy.cosmology import LambdaCDM
import astropy.units as u
from astropy.coordinates import SkyCoord
from scipy.interpolate import interp1d
import scipy.stats as stats
from astropy.stats import sigma_clip
c=const.c.to("km/s")


cosmos = LambdaCDM(H0=67.77* u.km / u.Mpc / u.s, Om0=0.307115, Ode0=0.692885)  # define cosmology on the basis of simulation

#the file
my_file=astropy.io.fits.open('Most_massive_MD04.fits')
my_file.info()

data= my_file[1].data
# Actual separation from the data
r=data.field('separation')
# Redshift  of the member galaxies
z=data.redshift_R_2
# Redshift at the center of the cluster
z_cluster=data.redshift_R_1[0]
ra=data.field('RA_2')
dec=data.field('DEC_2')
c1=ra*(np.pi/180)
c2=dec*(np.pi/180)
ra_cl=data.field('RA_1')[0]*np.pi/180
dec_cl=data.field('DEC_1')[0]*np.pi/180 #

# projecting the cluster 
plt.plot(dec,ra,color='black', linestyle='none', linewidth = 2, marker='.', markerfacecolor='blue', markersize=12)
plt.xlabel('Dec')
plt.ylabel('RA')

# Angular separation between two galaxy members inside the cluster.
#by using haversine method
ra_gal = c1     # ra of the the galaxies in radian
dec_gal = np.pi/2. -c2  #dec of a galaxies in radian
ra_cl = ra_cl
dec_cl = np.pi/2. - (dec_cl)
y = 2*np.arcsin(np.sqrt(np.sin((dec_cl-dec_gal)/2.0)**2.0 +np.sin(dec_cl)*np.sin(dec_gal)*np.sin((ra_cl-ra_gal)/2.0)**2.0))

# From astropy inbuilt module separation
A=(SkyCoord(data.field('RA_2')*u.degree, data.field('DEC_2')*u.degree)) #c1 refers to the galaxies coordinates
B=(SkyCoord(data.field('RA_1')[0]*u.degree,data.field('DEC_1')[0]*u.degree)) #c2 refers to the co-ordinates of whole cluster
sep = A.separation(B)
x=sep.radian

# Transverse comoving distance
D=cosmos.comoving_distance(z_cluster) #angular diameter distance
D_cl=cosmos.comoving_distance(z)
# Proper comoving distance
D_prop=cosmos.angular_diameter_distance(z_cluster)

# The projected radius of the cluster
r_proj=(x*D) # D is comoving distance
r_proj

# Getting the actual distance from the simulation
""" For the whole members of the cluster he actual estimation
"""

d_C = D_cl
dc_mpc = (d_C).value
dc_interpolation = interp1d(z, dc_mpc)
z_interpolation = interp1d(dc_mpc, z)

phi   = ( ra   - 180 ) * np.pi / 180.
theta = (dec + 90 ) * np.pi / 180.
rr    = dc_interpolation(z)
xx = rr * np.cos( phi) * np.sin( theta )
yy = rr * np.sin( phi) * np.sin( theta )
zz = rr * np.cos( theta )

""" For the cluster itself"""
d_x = cosmos.comoving_distance(data.redshift_R_1[0])
dx_mpc = (d_x).value
r1=dx_mpc
phi1   = ( data.field('RA_1')[0]   - 180 ) * np.pi / 180.
theta1 = (data.field('DEC_1')[0] + 90 ) * np.pi / 180.
x1 = r1 * np.cos( phi1) * np.sin( theta1 )
y1 = r1 * np.sin( phi1) * np.sin( theta1 )
z1 = r1 * np.cos( theta1 )

# The distance estimation
dis =np.sqrt((x1-xx)**2 + (y1-yy)**2 + (z1-zz)**2)
plt.plot(dis,r)

#Proj_radius
N,R=np.histogram(np.array(r_proj))
# Actual radius
nn,rs=np.histogram(np.array(dis))
rs

#Getting the number density profile in 2D 
n=N/(np.pi*(( R[1:]**2-R[:-1]**2)))

# central point of each separation 
R_proj = (R[1:] + R[:-1])/2
plt.plot(R_proj,n, color='black', linestyle='dashed', linewidth = 2, marker='o', markerfacecolor='green', markersize=12)
#plt.yscale('log')
#plt.xscale('log')
plt.xlabel('projected separation(Mpc)')
plt.ylabel('2D number density (Mpc^-2)')  
plt.show()

#Getting the derivative term of n wrt R_proj  ie dndR
dndR=np.array(np.gradient(n,R_proj))

"""
The interplation of dndR for solving the integral first of all 
Stacking the Proj and dndR

""" 
xx=np.hstack((np.array([0]),np.array(R_proj)))
yy=np.hstack((np.array(dndR[0]),np.array(dndR)))

# getting the interpolation
inter1=interp1d(xx, yy, bounds_error=False, fill_value=-10.)

# De-projection of the number density profile from 
# 2D to 3D using the Abel inversion equation

def nu(R,rs):
    return (-1/np.pi)*(inter1(R)/((R**2 - rs**2)**0.5))

nu_all=[integrate.quad(nu, rs_i, 1.6,args=(rs_i))[0] for rs_i in rs]

# plotting the de-projected number density wrt actual distance from the cluster center

plt.plot(rs,nu_all, color='black', linestyle='dashed', linewidth = 2, marker='o', markerfacecolor='green', markersize=12)
plt.xlabel('actual distance from the cluster center')
plt.ylabel('De-projected number dneisty')
plt.show()


""" The velocity profile of the cluster can be given as
"""

# Peculiar velocities of a members from there spectrosocpic
# redshift and mean redshift of a cluster

c=const.c.to("km/s")
z_cl=z_cluster
los_v=( c*z -  c*z_cl)/(1 + z_cl)
los_v
plt.plot(r,los_v, color='black', linestyle='none', linewidth = 2, marker='o', markerfacecolor='green', markersize=12)
plt.xlabel('Separation in MPC')
plt.ylabel('Rest frame velocities around the cluster')
plt.show()

""" from the biweight estimator one can retriv the velocities from the 
# Rest frame velocities 
 and the significant error
"""
val= biweight_scale(los_v, 9)
val_err = 0.92 * val / (np.sqrt(los_v.size - 1))
val,val_err

# Further in order to remove the interlopers  the 3 sigma clipping is used 
filtered_data = sigma_clip(los_v, sigma=2.5, maxiters=10000)

#clipped=sigma_clipped_stats(los_v, sigma=3, maxiters=1000)


# plot the original and rejected data
plt.figure(figsize=(10,8))

plt.plot(r_proj,los_v, '+', color='#1f77b4', label="original data")
plt.plot(r_proj[filtered_data.mask], los_v[filtered_data.mask], 'x',color='#d62728', label="rejected data")
plt.xlabel('Projected distance')
plt.ylabel('Rest frame velocities')
plt.legend(loc=1, numpoints=1)
