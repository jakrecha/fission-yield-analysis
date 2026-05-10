"""
Analysis of neutron-induced fission yield distributions
for U-233 and U-235 using TALYS-generated data.

The project includes:
- double-Gaussian fitting of fragment mass distributions,
- comparison of thermal fission yields,
- analysis of long-lived fission product production,
- energy dependence of fitted parameters.

Author: Jakub Krysinski
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import glob

#reading yield
def read_yield(file):
    
    """
    Read TALYS fission yield data file.

    The function extracts incident neutron energy from
    the file header and loads numerical yield data.

    Parameters
    ----------
    file : str
        Path to TALYS yield file.

    Returns
    -------
    energy : float
        Incident neutron energy in MeV.
    data : ndarray
        Numerical yield data loaded from file.
    """
    
    with open(file) as f:
        for line in f:
            if line.startswith("#   E-incident [MeV]: "):
                energy = float(line.split()[3])
                break
    data = np.loadtxt(file, comments='#')
    return energy, data

    
    
#defining sum of to gaussians
def double_gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2):
    
    """
    Calculate double-Gaussian approximation of fission yield distribution.

    The model represents asymmetric fission fragment mass
    distribution as a sum of two Gaussian peaks corresponding
    to light and heavy fission fragments.

    Parameters
    ----------
    x : ndarray
        Mass number values.
    A1 : float
        Amplitude of light fragment peak.
    mu1 : float
        Mean mass number of light fragment peak.
    sigma1 : float
        Standard deviation of light fragment peak.
    A2 : float
        Amplitude of heavy fragment peak.
    mu2 : float
        Mean mass number of heavy fragment peak.
    sigma2 : float
        Standard deviation of heavy fragment peak.

    Returns
    -------
    ndarray
        Double-Gaussian distribution values.
    """
    return A1*np.exp(-(x-mu1)**2/(2*sigma1**2)) + A2*np.exp(-(x-mu2)**2/(2*sigma2**2))

def fit_plot(base_dir, isotope):
    
    """
    Fit double-Gaussian model to TALYS fission yield distributions.

    The function:
    - loads yield distributions for different neutron energies,
    - performs nonlinear least-squares fitting,
    - generates comparison plots between TALYS data and fitted model,
    - stores fitted parameters for further analysis.

    Parameters
    ----------
    base_dir : str
        Base directory of the project.
    isotope : str
        Fissile isotope identifier (e.g. 'U233', 'U235').

    Returns
    -------
    fits : ndarray
        Array containing:
        [energy, A1, mu1, sigma1, A2, mu2, sigma2]
        for each fitted energy point.
    """
    path = os.path.join(base_dir, '..', isotope, 'output','yieldA*.fis')
    save_path = os.path.join(base_dir, '..', 'figures', 'fy'+isotope+'.png')
    files = glob.glob(path)
    fits= np.zeros((len(files), 7))
    #plt.figure()
    plt.figure(figsize=(12,18))
    for i, file in enumerate(files):
        energy, data = read_yield(file)
        #setting aid parameters for better fit
        p0 = [np.max(data[:,4]), 100, 5, np.max(data[:,4]), 130, 5]
        params, cov = curve_fit(double_gaussian, data[50:-50,0], data[50:-50,4], p0=p0, absolute_sigma=True) #fitting with reduced range to reduce tails impact
        #making sure sigmas are positive
        params[2]=abs(params[2])
        params[5]=abs(params[5])
        fit = double_gaussian(X_iter, *params)
        plt.subplot(5,2,i+1)
        plt.scatter(data[:,0], data[:,4], marker='^', linewidths=.1, label = 'TALYS data')
        plt.plot(X_iter, fit, label = f'fit')
        #showing fit parameters
        plt.text(params[1], params[0] * 0.5,
            f'$\\mu_1 = {params[1]:.2f}$\n$\\sigma_1 = {params[2]:.2f}$\n$A_1= {params[0]:.2f}$', ha='center', va='top',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
        plt.text(params[4], params[3] * 0.5,
            f'$\\mu_2 = {params[4]:.2f}$\n$\\sigma_2 = {params[5]:.2f}$\n$A_2= {params[3]:.2f}$', ha='center', va='top',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
        #adjusting plot limits with sigmas    
        plt.xlim(params[1]-5*params[2], params[4]+5*params[5])
        plt.legend()
        plt.xlabel('Mass number A')
        plt.ylabel(f'Cross section $\\sigma_x$ [mb]')
        plt.title(f'{energy} MeV')
        fits[i,0] = energy
        fits[i,1:] = params
    plt.tight_layout()
    plt.savefig(save_path)
    return fits

if __name__ == "__main__":
    #setting base directory
    base_dir = os.path.dirname(__file__)
    thermal_energy = 1.0e-06
    X_iter = np.linspace(0, 200, 2001) 
    U233_fit = fit_plot(base_dir, 'U233')
    U235_fit = fit_plot(base_dir, 'U235')


    mask = (U233_fit[:, 0] == thermal_energy )
    U233_thermal = U233_fit[mask]
    U233_thermal = U233_thermal[0]
    U233_thermal_fit = double_gaussian(X_iter, *U233_thermal[1:])

    mask = (U235_fit[:, 0] == thermal_energy )
    U235_thermal = U235_fit[mask]
    U235_thermal = U235_thermal[0]
    U235_thermal_fit = double_gaussian(X_iter, *U235_thermal[1:])

    #Plotting fits for 2 nuclides for thermal energy
    plt.figure()
    plt.plot(X_iter, U233_thermal_fit, label = 'U-233 fit', color = 'blue')
    plt.plot(X_iter, U235_thermal_fit, label = 'U-235 fit', color = 'red')
    plt.annotate(f'$\\mu_1 = {U233_thermal[2]:.2f}$\n$\\sigma_1 = {U233_thermal[3]:.2f}$\n$A_1= {U233_thermal[1]:.2f}$', xy=(U233_thermal[2], max(U233_thermal_fit)),
                xytext=(U233_thermal[2]-U233_thermal[3], max(U233_thermal_fit)) , ha = 'right', va = 'top', arrowprops=dict(arrowstyle='->', color = 'blue'),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.7))
    plt.annotate(f'$\\mu_1 = {U233_thermal[5]:.2f}$\n$\\sigma_1 = {U233_thermal[6]:.2f}$\n$A_1= {U233_thermal[4]:.2f}$', xy=(U233_thermal[5], max(U233_thermal_fit)),
                xytext=(U233_thermal[5]-U233_thermal[6], max(U233_thermal_fit)*.7) , ha = 'right', va = 'top', arrowprops=dict(arrowstyle='->', color = 'blue'),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.7))

    plt.annotate(f'$\\mu_2 = {U235_thermal[2]:.2f}$\n$\\sigma_2 = {U235_thermal[3]:.2f}$\n$A_2= {U235_thermal[1]:.2f}$', xy=(U235_thermal[2], max(U235_thermal_fit)),
                xytext=(U235_thermal[2]+U235_thermal[3], max(U235_thermal_fit)) , ha = 'left', va = 'top', arrowprops=dict(arrowstyle='->', color = 'red'),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.7))
    plt.annotate(f'$\\mu_2 = {U235_thermal[5]:.2f}$\n$\\sigma_2 = {U235_thermal[6]:.2f}$\n$A_2= {U235_thermal[4]:.2f}$', xy=(U235_thermal[5], max(U235_thermal_fit)),
                xytext=(U235_thermal[5]+U235_thermal[6], max(U235_thermal_fit)*.7) , ha = 'left', va = 'top', arrowprops=dict(arrowstyle='->', color = 'red'),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.7))

    plt.xlim(U235_thermal[2]-5*U235_thermal[3], U235_thermal[5]+5*U235_thermal[6])
    plt.xlabel('Mass number A')
    plt.ylabel(f'Cross section $\\sigma_x$ [mb]')
    plt.title(f'Comparison between U-233 and U-235 fission yield for {thermal_energy} MeV')
    plt.legend()
    plt.savefig(os.path.join(base_dir, '..', 'figures', 'Thermal_comparison.png'))

    #Plotting cs vs energy for both nuclides
    plt.figure()
    plt.scatter(U233_fit[:,0], U233_fit[:,1], marker='^', linewidths=.5, edgecolors='black', alpha=.7, label = f'$A_1$ for U-233')
    plt.scatter(U235_fit[:,0], U235_fit[:,1], marker='v', linewidths=.5, edgecolors='black', alpha=.7, label = f'$A_1$ for U-235', color = 'red')
    plt.legend()
    plt.xlabel('Energy [Mev]')
    plt.ylabel(f'Fit parameter $A_1$ [mb]')
    plt.loglog()
    plt.grid()
    plt.title(f'Cross section of fission reaction vs energy')
    plt.savefig(os.path.join(base_dir, '..', 'figures', 'cs_vs_en.png'))


    #yield Tc99 43, Cs137 55, I129 53
    path_Tc99_3 = os.path.join(base_dir, '..', 'U233', 'output','rp043099.tot')
    Tc99_3 = np.loadtxt(path_Tc99_3, comments='#')
    Tc99_3[Tc99_3 == 0] = 10e-12 #changing zeros to small values for log yaxis 
    path_Tc99_5 = os.path.join(base_dir, '..', 'U235', 'output','rp043099.tot')
    Tc99_5 = np.loadtxt(path_Tc99_5, comments='#')
    Tc99_5[Tc99_5 == 0] = 10e-12 #changing zeros to small values for log yaxis 
    plt.figure(figsize=(12,18))
    plt.subplot(3,1,1)
    plt.scatter(Tc99_3[:,0],Tc99_3[:,1], marker='^', linewidths=.5, edgecolors='black', alpha=.7, label='Tc-99 from U-233')
    plt.scatter(Tc99_5[:,0],Tc99_5[:,1], marker='v', linewidths=.5, edgecolors='black', alpha=.7, label='Tc-99 from U-235', color = 'red')
    plt.grid()
    plt.loglog()
    plt.legend()
    plt.ylabel('Effective production cross section [mb]')
    plt.title('Tc-99')
    plt.xlabel('Energy [Mev]')

    #Cs137
    path_Cs137_3 = os.path.join(base_dir, '..', 'U233', 'output','rp055137.tot')
    Cs137_3 = np.loadtxt(path_Cs137_3, comments='#')
    Cs137_3[Cs137_3 == 0] = 10e-12 #changing zeros to small values for log yaxis 
    path_Cs37_5 = os.path.join(base_dir, '..', 'U235', 'output','rp055137.tot')
    Cs137_5 = np.loadtxt(path_Cs37_5, comments='#')
    Cs137_5[Cs137_5 == 0] = 10e-12 #changing zeros to small values for log yaxis 
    plt.subplot(3,1,2)
    plt.scatter(Cs137_3[:,0],Cs137_3[:,1], marker='^', linewidths=.5, edgecolors='black', alpha=.7, label='Cs-137 from U-233')
    plt.scatter(Cs137_5[:,0],Cs137_5[:,1], marker='v', linewidths=.5, edgecolors='black', alpha=.7, label='Cs-137 from U-235', color = 'red')
    plt.grid()
    plt.loglog()
    plt.legend()
    plt.ylabel('Effective production cross section [mb]')
    plt.xlabel('Energy [Mev]')
    plt.title('Cs-137')

    #I129
    path_I_129_3 = os.path.join(base_dir, '..', 'U233', 'output','rp053129.tot')
    I129_3 = np.loadtxt(path_I_129_3, comments='#')
    I129_3[I129_3 == 0] = 10e-12 #changing zeros to small values for log yaxis 
    path_I_129_5 = os.path.join(base_dir, '..', 'U235', 'output','rp053129.tot')
    I129_5 = np.loadtxt(path_I_129_5, comments='#')
    I129_5[I129_5 == 0] = 10e-12 #changing zeros to small values for log yaxis 
    plt.subplot(3,1,3)
    plt.scatter(I129_3[:,0],I129_3[:,1], marker='^', linewidths=.5, edgecolors='black', alpha=.7, label='I-129 from U-233')
    plt.scatter(I129_5[:,0],I129_5[:,1], marker='v', linewidths=.5, edgecolors='black', alpha=.7, label='I-129 from U-235', color = 'red')
    plt.grid()
    plt.loglog()
    plt.legend()
    plt.ylabel('Effective production cross section [mb]')
    plt.xlabel('Energy [Mev]')
    plt.title('I-129')

    plt.tight_layout()
    save_path_nuclides = os.path.join(base_dir, '..', 'figures', 'nuclides.png')
    plt.savefig(save_path_nuclides)

