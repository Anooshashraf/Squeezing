"""
BBO Crystal Squeezing Simulation
"""
import numpy as np
from scipy import constants, integrate, interpolate
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class BBOSqueezer:
    """
    Simulate squeezing generation from BBO crystal
    """
    
    def __init__(self):
        # Physical constants
        self.hbar = constants.hbar
        self.c = constants.c
        self.eps0 = constants.epsilon_0
        
        # BBO crystal parameters (β-BaB2O4)
        self.bboparams = {
            'd11': 2.0e-12,          # Nonlinear coefficient (m/V)
            'n_o_532': 1.676,        # Ordinary index @ 532 nm
            'n_e_532': 1.557,        # Extraordinary index @ 532 nm
            'n_o_1064': 1.655,       # Ordinary index @ 1064 nm
            'n_e_1064': 1.542,       # Extraordinary index @ 1064 nm
            'alpha': 0.001,          # Absorption coefficient (1/cm)
            'dn_dT': 1.0e-5,         # Thermo-optic coefficient (1/K)
            'walkoff_angle': 3.0,    # Walk-off angle (degrees)
            'damage_threshold': 10e9 # Damage threshold (W/m²)
        }
    
    def phase_matching_curve(self, temperature_range=(20, 40), 
                             crystal_length=5e-3, pump_power=100e-3):
        """
        Calculate phase matching vs temperature
        """
        temperatures = np.linspace(temperature_range[0], 
                                  temperature_range[1], 100)
        
        # Refractive indices vs temperature (Sellmeier equation simplified)
        n_pump = self.bboparams['n_e_532'] + self.bboparams['dn_dT'] * (temperatures - 25)
        n_signal = self.bboparams['n_o_1064'] + self.bboparams['dn_dT'] * (temperatures - 25)
        n_idler = n_signal  # Degenerate case
        
        # Phase mismatch (Type I: o + o → e)
        lambda_pump = 532e-9
        lambda_signal = 1064e-9
        
        delta_k = 2*np.pi * (n_pump/lambda_pump - n_signal/lambda_signal - n_idler/lambda_signal)
        
        # Sinc^2 phase matching function
        sinc_term = (np.sinc(delta_k * crystal_length / (2*np.pi)))**2
        
        # Convert to efficiency
        efficiency = sinc_term * np.exp(-self.bboparams['alpha'] * crystal_length * 100)
        
        return temperatures, efficiency, delta_k
    
    def calculate_squeezing(self, pump_power, crystal_length, temperature=25,
                           method='single_pass', cavity_params=None):
        """
        Calculate squeezing level from BBO
        
        Parameters:
        -----------
        pump_power : float
            Pump power in Watts
        crystal_length : float
            Crystal length in meters
        temperature : float
            Crystal temperature in °C
        method : str
            'single_pass' or 'OPO'
        cavity_params : dict
            For OPO: {'finesse': 100, 'loss': 0.01, 'output_coupling': 0.1}
        """
        
        # Calculate parametric gain
        # Simplified gain calculation
        L = crystal_length
        P_pump = pump_power
        
        # Nonlinear gain coefficient
        d_eff = self.bboparams['d11']
        omega_p = 2*np.pi*self.c/532e-9
        n_p = self.bboparams['n_e_532']
        n_s = self.bboparams['n_o_1064']
        
        # Effective area (assuming beam waist 50 μm)
        A_eff = np.pi * (50e-6)**2
        
        # Coupling coefficient
        gamma = d_eff * omega_p / (self.c * np.sqrt(n_p * n_s * self.eps0 * A_eff))
        
        # Single-pass gain
        g0 = gamma * np.sqrt(P_pump) * L
        
        if method == 'single_pass':
            # Single-pass squeezing
            squeezing_db = 10 * np.log10(np.exp(-2 * g0))
            bandwidth = self.c / (2 * n_s * L)  # Rough estimate
            
        elif method == 'OPO' and cavity_params:
            # OPO cavity squeezing
            finesse = cavity_params.get('finesse', 100)
            loss = cavity_params.get('loss', 0.01)
            T_out = cavity_params.get('output_coupling', 0.1)
            
            # Threshold pump power
            P_th = (loss + T_out)**2 / (4 * gamma**2 * L**2)
            
            # Pump parameter
            sigma = np.sqrt(P_pump / P_th)
            
            # OPO squeezing (below threshold)
            squeezing_db = 10 * np.log10(1 - sigma**2)
            
            # Cavity bandwidth
            FSR = self.c / (2 * n_s * L)  # Free spectral range
            bandwidth = FSR / finesse
            
        else:
            raise ValueError("Invalid method or missing cavity_params")
        
        # Apply temperature dephasing
        temp_offset = np.abs(temperature - 25)  # Reference at 25°C
        dephasing = np.exp(-(temp_offset/5)**2)  # Gaussian dephasing
        
        squeezing_db *= dephasing
        
        return {
            'squeezing_db': squeezing_db,
            'bandwidth': bandwidth,
            'gain': g0,
            'method': method
        }
    
    def simulate_walkoff(self, crystal_length, beam_waist=50e-6):
        """
        Simulate walk-off effects in BBO
        """
        walkoff_rad = np.radians(self.bboparams['walkoff_angle'])
        
        # Walk-off distance
        walkoff_distance = crystal_length * np.tan(walkoff_rad)
        
        # Overlap efficiency
        overlap_efficiency = np.exp(-(walkoff_distance/beam_waist)**2)
        
        return {
            'walkoff_distance': walkoff_distance,
            'overlap_efficiency': overlap_efficiency,
            'critical_length': beam_waist / np.tan(walkoff_rad)
        }
    
    def plot_simulation_results(self):
        """
        Generate comprehensive plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Phase matching vs temperature
        temps, eff, dk = self.phase_matching_curve()
        axes[0,0].plot(temps, eff, 'b-', linewidth=2)
        axes[0,0].set_xlabel('Temperature (°C)')
        axes[0,0].set_ylabel('Phase Matching Efficiency')
        axes[0,0].set_title('BBO Phase Matching vs Temperature')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Squeezing vs pump power
        pump_powers = np.linspace(10e-3, 500e-3, 50)  # 10-500 mW
        squeezing_single = []
        squeezing_opo = []
        
        for P in tqdm(pump_powers, desc="Calculating squeezing"):
            # Single pass
            res_single = self.calculate_squeezing(P, 5e-3, method='single_pass')
            squeezing_single.append(res_single['squeezing_db'])
            
            # OPO
            cavity_params = {'finesse': 100, 'loss': 0.01, 'output_coupling': 0.1}
            res_opo = self.calculate_squeezing(P, 5e-3, method='OPO', 
                                             cavity_params=cavity_params)
            squeezing_opo.append(res_opo['squeezing_db'])
        
        axes[0,1].plot(pump_powers*1000, squeezing_single, 'r-', label='Single Pass')
        axes[0,1].plot(pump_powers*1000, squeezing_opo, 'b-', label='OPO')
        axes[0,1].set_xlabel('Pump Power (mW)')
        axes[0,1].set_ylabel('Squeezing (dB)')
        axes[0,1].set_title('Squeezing vs Pump Power')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Walk-off effects
        lengths = np.linspace(1e-3, 20e-3, 50)  # 1-20 mm
        efficiencies = []
        
        for L in lengths:
            walkoff = self.simulate_walkoff(L)
            efficiencies.append(walkoff['overlap_efficiency'])
        
        axes[1,0].plot(lengths*1000, efficiencies, 'g-', linewidth=2)
        axes[1,0].set_xlabel('Crystal Length (mm)')
        axes[1,0].set_ylabel('Overlap Efficiency')
        axes[1,0].set_title('Walk-off Effects vs Crystal Length')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Damage threshold check
        intensities = pump_powers / (np.pi * (50e-6)**2)
        safe_mask = intensities < self.bboparams['damage_threshold']
        
        axes[1,1].plot(pump_powers*1000, intensities/1e9, 'm-', linewidth=2)
        axes[1,1].fill_between(pump_powers*1000, 0, self.bboparams['damage_threshold']/1e9,
                              alpha=0.3, color='green', label='Safe Zone')
        axes[1,1].axhline(y=self.bboparams['damage_threshold']/1e9, 
                         color='red', linestyle='--', label='Damage Threshold')
        axes[1,1].set_xlabel('Pump Power (mW)')
        axes[1,1].set_ylabel('Intensity (GW/m²)')
        axes[1,1].set_title('Damage Threshold Check')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('bbo_simulation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig