import numpy as np
import astropy.io.fits
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy import log,exp,sqrt,stats
import scipy as sp
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
import sherpa.astro.ui as ui
from colossus.cosmology import cosmology
from colossus.halo import profile_nfw
cosmology.setCosmology('planck18')

#define cosmology
cosmo = LambdaCDM(H0=67.77* u.km / u.Mpc / u.s, Om0=0.307115, Ode0=0.692885)

#speed of light
c=const.c.to("km/s")

# opening the file+
path_2_data = 'Most_massive_MD04.fits'
print('opening', path_2_data)
my_file=astropy.io.fits.open(path_2_data)
#print(my_file[1].data.columns)
data = my_file[1].data

# coordinates of the given Halo
z_cl = data.field('redshift_S_1')[0] # redshift
ra_cl = data.field('RA_1')[0]*np.pi/180 # radians
dec_cl= data.field('DEC_1')[0]*np.pi/180 # radians
r_cl = data.field('HALO_Rvir_1')[0] # virial radii

# coordinates of the sub haloes
z   = data.field('redshift_S_2') # redshift
ra  = data.field('RA_2') * (np.pi/180) # radians
dec = data.field('DEC_2') * (np.pi/180) # radians

# 3D separation in Mpc, to be verified
r = data.field('separation') # Mpc

#Scale radius of the halo
rs=data.field('HALO_rs_2')/1000
r_scale=rs.max()

# Redshift histogram
vx = plt.hist(data.field('redshift_S_2'))
plt.show()

#plt.hist(data.field('redshift_R_2')) #simulated redshift
#fig.savefig("hist")
# plottting the halo subhaloes projection
plt.plot(dec,ra, color='black', linestyle='none', linewidth = 2, marker='.', markerfacecolor='blue', markersize=12)
plt.plot(dec_cl,ra_cl, color='black', linestyle='dashed', linewidth = 2, marker='*', markerfacecolor='Red', markersize=12)
plt.show()

# 3d distance between halo and subhaloes
c_subhaloes = SkyCoord(ra=ra*180/np.pi*u.degree, dec=dec*180/np.pi*u.degree, distance=cosmo.comoving_distance(z).value*u.mpc)
c_cluster = SkyCoord(ra_cl*180/np.pi*u.degree, dec_cl*180/np.pi*u.degree, distance=cosmo.comoving_distance(z_cl).value*u.mpc)

# the cartesian position of the objects in x,y,z components using astropy
c_subhaloes.cartesian.x ,c_subhaloes.cartesian.y,c_subhaloes.cartesian.z
c_cluster.cartesian.x,c_cluster.cartesian.y,c_cluster.cartesian.z

# 3d distance estimation
distance=np.sqrt((c_cluster.cartesian.x-c_subhaloes.cartesian.x)**2 + \
    (c_cluster.cartesian.y-c_subhaloes.cartesian.y)**2 + (c_cluster.cartesian.z-c_subhaloes.cartesian.z)**2)
print('the minimum and maximum distance in between the Halo and subhaloes', np.min(distance), np.max(distance))

# projected distance
# From astropy inbuilt module separation
coordinates_SubHaloes = SkyCoord( ra * 180/np.pi, dec * 180/np.pi, unit='deg', frame='icrs')
coordinate_cluster    = SkyCoord( ra_cl * 180/np.pi, dec_cl*180/np.pi, unit='deg', frame='icrs')
Angular_separation_Astropy = coordinates_SubHaloes.separation( coordinate_cluster )
Angular_separation_Astropy_radian = (Angular_separation_Astropy).to(u.radian)

# Comoving distance
D = cosmo.comoving_distance(z_cl)
print('the cluster at redshift ',z_cl,' is at dC=',D)

# Angular diameter distance
print('Angular diameter distance at the clusters redshift :' , cosmo.angular_diameter_distance(z_cl), 'per radian')
print('Angular diameter distance at the clusters redshift :' , cosmo.angular_diameter_distance(z_cl)/(180/np.pi), 'per degree')
print('Angular diameter distance at the clusters redshift :' , \
      (cosmo.angular_diameter_distance(z_cl)/(180/np.pi)).to(u.kpc)/60, 'per arc minute    ')

# The angular separation between sub haloes and the cluster converted in Mpc
r_proj =  Angular_separation_Astropy_radian * cosmo.angular_diameter_distance(z_cl)/u.radian  # where D is comoving distance
print('min, max projected distance : ',r_proj.min(), r_proj.max(), 'compared to the 3D virial radius', r_cl/1000)

# poission distribution error with 68 % confidence

def poiss_err(n, alpha=0.32):
    """
    Poisson error (variance) for n counts.
    http://pdg.lbl.gov/2018/reviews/rpp2018-rev-statistics.pdf,

    :param: alpha corresponds to central confidence level 1-alpha,
            i.e. alpha = 0.32 corresponds to 68% confidence
    """
    sigma_lo = sp.stats.chi2.ppf(alpha/2,2*n)/2
    sigma_up = sp.stats.chi2.ppf(1-alpha/2,2*(n+1))/2
    return sigma_lo, sigma_up

# get the 2d / projected number density profile
nbins=10 # defining the bins
n_data,R_bins=np.histogram(r_proj.value, nbins, range=(r_proj.min().value, r_proj.max().value))
# ignoring the bins with zero counts
select = n_data > 0
n_data = n_data[select]
R_bins_low = R_bins[:-1][select]
R_bins_high = R_bins[1:][select]

# calculation of the poisson error in data set
n_data_low, n_data_high = poiss_err(n_data, alpha=0.32)
R_bins_ce = 0.5 * (R_bins_low + R_bins_high)

# projected density profile
radial_diff = (R_bins_high**2 - R_bins_low**2) # difference between higher and lower radial bin
Sig_data = n_data / (np.pi * radial_diff)
Sig_data_low = n_data_low / (np.pi * radial_diff)
Sig_data_high = n_data_high / (np.pi * radial_diff),

# error in the measurement ie. x and y error
sig_low = Sig_data - Sig_data_low
sig_high = Sig_data_high - Sig_data

# plot the projected number density wrt the radial distance from the center of the cluster
plt.figure(0, (10, 10))
plt.errorbar((R_bins_ce), Sig_data,yerr=(Sig_data - Sig_data_low),
    color='blue',  fmt='.', capsize=10, label='Data')
plt.plot(R_bins_ce,Sig_data,'o',linestyle='-')
plt.title('Number density of sub haloes around the host cluster')
plt.xlabel('Cluster-centric Separation  [Mpc]',fontsize=18)
plt.ylabel(r'number density [Mpc$^{-2}$]',fontsize=18)
plt.axvline(x=r_cl/1000.,linestyle='dashed',label = 'Rvir')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.show()



# similarly to get the direct estimate of the number density using 3d distances
n_data_,R_bins_=np.histogram(r, nbins, range=(r.min(), r.max())) # r is the 3d distance
#  repeating the same procedure as above
select = n_data_> 0
n_data_ = n_data_[select]
R_bins_low = R_bins_[:-1][select]
R_bins_high = R_bins_[1:][select]
#  compute the 68 % confidence in the distribution
n_data_low, n_data_high = poiss_err(n_data_, alpha=0.32)
r_mid = 0.5 * (R_bins_low + R_bins_high)

# direct estimate of number density
r_diff = (R_bins_high**3 - R_bins_low**3)
number_density = n_data_ /(4/3* (np.pi * r_diff))
n_den_low = n_data_low / (np.pi * r_diff)
n_den_high = n_data_high / (np.pi * r_diff), R_bins_ce

# error
sig_low = number_density - n_den_low
sig_high =n_den_high - number_density
#return R_bins_low,R_bins_high,n_data_low, n_data_high, Sig_data, Sig_data_high, Sig_data_low

plt.figure(0, (10, 10))
plt.errorbar((r_mid), number_density,yerr=(number_density - n_den_low),color='blue',
             fmt= '.', capsize=10,label='Direct estimate',linestyle='-')

plt.loglog()
plt.show()

# integrating the function to get nu deprojected
"""
The interplation of dndR for solving the integral first of all
Stacking the Proj and dndR

"""
#Getting the derivative term of n wrt R_proj  ie dndR
dndR = np.array(np.gradient(Sig_data,R_bins_ce))
n_2D_no0 = n_data
n_2D_no0[n_data==0] = 1

dndR_up = np.array(np.gradient( Sig_data + Sig_data * n_2D_no0**-0.5, R_bins_ce ))
dndR_low = np.array(np.gradient( Sig_data - Sig_data * n_2D_no0**-0.5, R_bins_ce ))

def get_nu_all(dndR):
    xx = np.hstack((np.array([0]),np.array(R_bins_ce), np.array([10.]) ))
    yy = np.hstack((np.array(dndR[0]),np.array(dndR), np.array([0.]) ))

    # getting the interpolation
    inter1 = interp1d(xx, yy, bounds_error=True)

    # De-projection of the number density profile from
    # 2D to 3D using the Abel inversion equation

    def nu(R, r):
        """
        Latex equation
        Reference to the article it comes from
        """
        return (-1/np.pi)*(inter1(R)/((R**2 - r**2)**0.5))

    nu_all_diff=[integrate.quad(nu, r_i, 1.90, args=(r_i))[0] for r_i in r_mid]
    return nu_all_diff

nu_all_diff = get_nu_all(dndR)
nu_all_up = get_nu_all(dndR_up)
nu_all_low = get_nu_all(dndR_low)
nu_all_diff

# Using Beta
def Beta_nu(r,rhos,rs,beta):
    return rhos / (1.0 + (r/rs)** 2)**beta

#Curve fit
parameters, covariance = curve_fit(Beta_nu,r_mid, nu_all_diff)
fit_A=parameters[0]
fit_B=parameters[1]
fit_c=parameters[2]
#plot
fit_y = Beta_nu(r_mid, fit_A, fit_B,fit_c)
plt.figure(0, (10, 10))
plt.plot(r_mid, nu_all_diff, 'o', label='data')
plt.plot(r_mid, fit_y, '-', label='Beta_profile/ Kings')
plt.fill_between(r_mid, y1=nu_all_low, y2=nu_all_up, alpha=0.3, color='k')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.show()


#
# velocity dispersion

# rest frame distribution of the subhaloes wrt the parent halo
c=const.c.to("km/s")
los_v= c*(z - z_cl)/(1 + z_cl)
los_v.value

#Plotting the phase space distribution of the rest frame velocities wrt the radial distances
plt.figure(0, (10, 10))
plt.title('rest frame velocity distribution')
plt.plot(r_proj,los_v, color='black', linestyle='none', linewidth = 2, marker='.', markerfacecolor='green', markersize=12)
plt.xlabel('Separation in [Mpc]')
plt.ylabel('Peculiar velocity in clusters frame')
plt.show()


# histogram of the peculiar velocities
avg = np.mean(los_v.value)
var = np.var(los_v.value)
# From that, we know the shape of the fitted Gaussian.
pdf_x = np.linspace(np.min(los_v.value),np.max(los_v.value),100)
pdf_y = 1.0/np.sqrt(2*np.pi*var)*np.exp(-0.5*(pdf_x-avg)**2/var)
plt.figure()
plt.figure(0, (10, 10))
plt.hist(los_v.value,30,density=True)
plt.plot(pdf_x,pdf_y,'--')
plt.legend(("Fit","Data"),"best")
plt.show()

# defining the power law to fit the projected velocity dispersion profile
def sigma_v(r,v_0):
    beta=0.1
    return v_0 / (1+ r) ** beta

# biweight estimates of the velocity dispersion for further analysis
def biweight_vdisp(v):

    if v.size > 2:
        sigma = biweight_scale(v, 9)
        return sigma

 # error associated with it
#def biweight_est_error(v):
   # sigma_err = 0.92 * sigma / (np.sqrt(v.size - 1))
   # return sigma_err

# gapper estimate
def gapper_vdisp(v):

    v = np.sort(v)
    n = len(v)
    w = np.arange(1, n) * np.arange(n - 1, 0, -1)
    g = np.diff(v)
    return (np.sqrt(np.pi)) / (n * (n - 1)) * np.sum(w * g)


# function to get the velocity dispersion profile
def velocity_dispersion_profile(r, v, nbins, method="biweight"):

    x= r_proj.value
    y= los_v.value

    d, N, mean = [np.zeros(nbins) for i in range(3)]
    bin_width = (max(x) - min(x))/nbins
    left_bin_edges = np.array([min(x) + i*bin_width for i in range(nbins)])
    right_bin_edges = left_bin_edges + bin_width
    mid_bin = left_bin_edges + .5*bin_width

    for i in range(nbins):
        m = (left_bin_edges[i] < x) * (x < right_bin_edges[i])
        mean[i] = np.mean(y[m])
        N[i] = sum(m)
        if method == "std":
            d[i] = np.std(y[m])
        elif method == "mad":

            d[i] = np.sqrt(np.median(y[m]))
        elif method == "rms":
            d[i] = np.sqrt(np.mean((y[m])**2))
        elif method == "biweight":
            d[i] = biweight_vdisp((y[m]))
        elif method == "gapper":
            d[i] = gapper_vdisp((y[m]))

    return mid_bin, d, d/np.sqrt(N), mean

# getting the profile
bins, dbins, err, mean = velocity_dispersion_profile(r_proj.value, los_v.value, 8, method="biweight")
bins1, dbins1, err1, mean1=velocity_dispersion_profile(r_proj.value, los_v.value, 8, method="std")
bins2, dbins2, err2, mean2 =velocity_dispersion_profile(r_proj.value, los_v.value, 8, method="rms")
bins3, dbins3, err3, mean3 =velocity_dispersion_profile(r_proj.value, los_v.value, 8, method="gapper")


#plotting the Velocity dispersion profile
plt.figure(0, (10, 10))
plt.title("Velocity dispersion profile")
plt.errorbar(bins, dbins, yerr=err,color='red',   label="biweight")
plt.errorbar(bins1, dbins1, yerr=err1,color='blue', fmt="*",  label="std deviation")
plt.errorbar(bins2, dbins2, yerr=err2,color='green', fmt="^", label="rms")
plt.errorbar(bins3, dbins3, yerr=err3,fmt=".", label="gapper")


plt.plot(bins, dbins, "k", lw=.8)
plt.plot(bins1, dbins1, "k", lw=.8)
plt.plot(bins2, dbins2, "k", lw=.8)
plt.plot(bins3, dbins3, "k", lw=.8)



# fitting using the power law for the line of sight velocity dispersion profile
popt, pcov = curve_fit(sigma_v,bins,dbins)

# plotting the function
plt.figure(0, (10, 10))
plt.title("Velocity dispersion profile")

plt.plot(bins, sigma_v(bins, *popt), 'r-',label='fit' )
plt.xlabel("$\mathrm{r~[Mpc]}$")
plt.ylabel("$\mathrm{\sigma_{los}~[kms^{-1}]}$")

#plt.xscale('log')
#plt.yscale('log')
plt.legend()
plt.show()
popt

# binned_dispersion
def binned_dispersion(x, y, nbins, method="mad"):

    d, N, mean = [np.zeros(nbins) for i in range(3)]
    bin_width = (max(x) - min(x))/nbins
    left_bin_edges = np.array([min(x) + i*bin_width for i in range(nbins)])
    right_bin_edges = left_bin_edges + bin_width
    mid_bin = left_bin_edges + .5*bin_width

    for i in range(nbins):
        m = (left_bin_edges[i] < x) * (x < right_bin_edges[i])
        mean[i] = np.mean(y[m])
        N[i] = sum(m)
        if method == "std":
            d[i] = np.std(y[m])
        elif method == "mad":
    #d[i] = np.median(np.abs(y[m]))
            d[i] = np.sqrt(np.median(y[m]))
        elif method == "rms":
            d[i] = np.sqrt(np.mean((y[m])**2))
    return mid_bin, d, d/np.sqrt(N), mean

#velocity dispersion from HALO vx ,vy and vz

velocity_dispersion=np.sqrt((data.field('HALO_vx_1')-data.field('HALO_vx_2'))**2 +\
                            (data.field('HALO_vy_1')-data.field('HALO_vy_2'))**2 + (data.field('HALO_vz_1')-data.field('HALO_vz_2'))**2)

plt.plot(r,velocity_dispersion,'+')
plt.xlabel('radial-distance in [Mpc]')
plt.ylabel('velocity_dispersion from Halo_vx,vy,vz[ Km/s]')
plt.show()

# radial dispersion profie from halo vx,vy and vz
radial_pt, _radial_dispersion, vel_err, mean = binned_dispersion(r, velocity_dispersion, 7, method="rms")

plt.figure(0, (10, 10))
plt.title("Velocity dispersion profile")
plt.errorbar(radial_pt, _radial_dispersion, yerr=vel_err, fmt="o", markeredgecolor="k", label="Velocity dispersion profile")
plt.plot(radial_pt, _radial_dispersion, "k", lw=.8)
plt.xlabel("$\mathrm{r~[Mpc]}$")
plt.ylabel("$\mathrm{\sigma_{los}~[kms^{-1}]}$")
plt.legend()
plt.show()

# determination of radial velocity dispersion
nvd= n_2D*(dbins**2)
# deprojecting the line of sight  velocity dispersion

dndV=np.array(np.gradient(nvd,bins))

def get_sigma_all(dndV):
    aa = np.hstack((np.array([0]),np.array(bins), np.array([10.]) ))
    bb = np.hstack((np.array(dndV[0]),np.array(dndV), np.array([0.]) ))

    # getting the interpolation
    interpol1 = interp1d(aa, bb, bounds_error=True)

    # De-projection of the line of sight velocity profile from
    # 2D to 3D using the Abel inversion equation

    def sigma(R, r):

        return (interpol1(R)/((R**2 - r**2)**0.5))



    sigma_all=np.array([integrate.quad(sigma, r_i, 1.90, args=(r_i))[0] for r_i in R_3D])
    return sigma_all

sigma_all = get_sigma_all(-dndV)/((np.pi)*np.array(nu_all_diff))
sigma_all

radial_vd=np.sqrt(sigma_all)
radial_vd
plt.title('Radial velocity dispersion profile')
plt.plot(R_3D,radial_vd,'o',linestyle= '-')
plt.xlabel("$\mathrm{r~[Mpc]}$")
plt.ylabel("$\mathrm{\sigma_{radial}~[kms^{-1}]}$")
plt.show()

#finding the slop of the density as well as velocity profile
dln_nu=R_3D /nu_all_diff *(np.gradient(nu_all_diff,R_3D))
dln_sigma=R_3D/radial_vd*(np.gradient(radial_vd,R_3D))
dln_nu,dln_sigma

# mass profile
vel_disp=740*10**3 # Km to meters
radius=(r_cl/1000)*3.086e+22 # 1.1 Mpc      # maximum distance from t3333he cluster's centre
M=-(vel_disp**2)*radius/(const.G) *5.0279e-31   *(dln_nu +dln_sigma) #converted into solar masses unless the unit remains the sane
M
# plotting mass profile
plt.plot(R_3D,M,'o',linestyle= '-')
plt.xlabel("$\mathrm{r~[Mpc]}$")
plt.ylabel("$\mathrm{M_{r}~[M_{sun}]}$")
plt.yscale('log')
plt.show()
