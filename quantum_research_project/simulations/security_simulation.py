"""
SDI Security Protocol Simulation
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import special, linalg
from tqdm import tqdm

class SDIQRNGSimulator:
    """
    Simulate SDI-QRNG security and performance
    """
    
    def __init__(self):
        # Parameters from the research paper
        self.params = {
            'delta': 0.01536,          # Measurement precision
            'epsilon': 1e-6,           # Security parameter
            'oscillator_bits': 6,      # Effective bits of OSC
            'hashing_efficiency': 0.95, # Toeplitz hashing efficiency
            'sampling_rate': 200e6     # 200 MSamples/s
        }
    
    def entropic_uncertainty_bound(self, squeezing_db, LO_noise_fraction=0.05,
                                  electronic_noise_db=-13):
        """
        Calculate conditional min-entropy bound (Eq. 6 from paper)
        """
        # Convert noises
        electronic_noise = 10**(electronic_noise_db/10)
        
        # Anti-squeezed quadrature variance
        var_antisqueezed = 10**(squeezing_db/10)  # > 1
        
        # Total measured variance
        var_total = var_antisqueezed + LO_noise_fraction + electronic_noise
        
        # Max-entropy for check quadrature (squeezed)
        var_squeezed = 10**(-np.abs(squeezing_db)/10)  # < 1
        H_max = 0.5 * np.log2(2 * np.pi * np.e * var_squeezed * self.params['delta']**2)
        
        # Incompatibility term c(δq, δp)
        # Using approximation from paper
        c_term = (self.params['delta']**2) / (2 * np.pi)
        
        # Smooth min-entropy bound (Eq. 6)
        H_min_bound = -np.log2(c_term) - H_max
        
        # Account for untrusted LO noise
        trusted_fraction = 1 - LO_noise_fraction / var_total
        H_min_bound *= trusted_fraction
        
        # Account for electronic noise
        quantum_fraction = var_antisqueezed / var_total
        H_min_bound *= quantum_fraction
        
        return H_min_bound, H_max
    
    def simulate_complete_protocol(self, squeezing_db, LO_noise_fraction, 
                                  electronic_noise_db, N_measurements=1e6):
        """
        Simulate complete SDI-QRNG protocol
        """
        # 1. Calculate entropy bounds
        H_min, H_max = self.entropic_uncertainty_bound(
            squeezing_db, LO_noise_fraction, electronic_noise_db)
        
        # 2. Apply finite-size effects (Eq. 7 from paper)
        delta_term = 4 * np.sqrt(np.log2(2/self.params['epsilon']**2))
        delta_term *= np.log2(2**(1 + H_max/2) + 1)
        
        H_min_smooth = H_min - delta_term / np.sqrt(N_measurements)
        
        # 3. Account for ADC resolution
        effective_bits = min(self.params['oscillator_bits'], 
                           int(np.floor(H_min_smooth)))
        
        # 4. Calculate secure bits per sample
        secure_bits_per_sample = effective_bits * self.params['hashing_efficiency']
        
        # 5. Calculate final rate
        rate_bps = secure_bits_per_sample * self.params['sampling_rate']
        rate_mbps = rate_bps / 1e6
        
        results = {
            'H_min_bound': H_min,
            'H_min_smooth': H_min_smooth,
            'effective_bits': effective_bits,
            'secure_bits_per_sample': secure_bits_per_sample,
            'rate_mbps': rate_mbps,
            'parameters': {
                'squeezing_db': squeezing_db,
                'LO_noise_fraction': LO_noise_fraction,
                'electronic_noise_db': electronic_noise_db,
                'N_measurements': N_measurements
            }
        }
        
        return results
    
    def simulate_attack_scenarios(self, base_squeezing=3.5):
        """
        Simulate different attack scenarios
        """
        scenarios = {
            'Ideal': {'LO_noise': 0.01, 'elec_noise': -15},
            'Realistic': {'LO_noise': 0.05, 'elec_noise': -13},
            'LO_Attack': {'LO_noise': 0.30, 'elec_noise': -13},
            'Noisy_Detector': {'LO_noise': 0.05, 'elec_noise': -10},
        }
        
        results = {}
        
        print("="*60)
        print("SDI-QRNG SECURITY ANALYSIS")
        print("="*60)
        
        for name, params in scenarios.items():
            res = self.simulate_complete_protocol(
                base_squeezing, 
                params['LO_noise'], 
                params['elec_noise']
            )
            
            results[name] = res
            
            print(f"\nScenario: {name}")
            print(f"  Squeezing: {base_squeezing:.1f} dB")
            print(f"  LO noise fraction: {params['LO_noise']:.3f}")
            print(f"  Electronic noise: {params['elec_noise']} dB")
            print(f"  Secure bits/sample: {res['secure_bits_per_sample']:.3f}")
            print(f"  Final rate: {res['rate_mbps']:.1f} Mbps")
            print(f"  Security reduction: {100*(1-res['H_min_smooth']/res['H_min_bound']):.1f}%")
        
        return results
    
    def plot_performance_analysis(self):
        """
        Plot comprehensive performance analysis
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Rate vs squeezing level
        squeezing_levels = np.linspace(0, 8, 50)
        rates_ideal = []
        rates_realistic = []
        
        for sq in tqdm(squeezing_levels, desc="Calculating rates"):
            # Ideal
            res_ideal = self.simulate_complete_protocol(sq, 0.01, -15)
            rates_ideal.append(res_ideal['rate_mbps'])
            
            # Realistic
            res_real = self.simulate_complete_protocol(sq, 0.05, -13)
            rates_realistic.append(res_real['rate_mbps'])
        
        axes[0,0].plot(squeezing_levels, rates_ideal, 'b-', label='Ideal (Low Noise)', linewidth=2)
        axes[0,0].plot(squeezing_levels, rates_realistic, 'r-', label='Realistic BBO', linewidth=2)
        axes[0,0].axvline(x=3.5, color='g', linestyle='--', label='Target (3.5 dB)')
        axes[0,0].set_xlabel('Squeezing Level (dB)')
        axes[0,0].set_ylabel('Secure Rate (Mbps)')
        axes[0,0].set_title('Secure Rate vs Squeezing Level')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Security vs LO noise
        LO_noises = np.logspace(-3, 0, 50)  # 0.001 to 1
        secure_bits = []
        
        for noise in LO_noises:
            res = self.simulate_complete_protocol(3.5, noise, -13)
            secure_bits.append(res['secure_bits_per_sample'])
        
        axes[0,1].semilogx(LO_noises, secure_bits, 'm-', linewidth=2)
        axes[0,1].axvline(x=0.05, color='r', linestyle='--', label='Realistic LO noise')
        axes[0,1].set_xlabel('LO Noise Fraction')
        axes[0,1].set_ylabel('Secure Bits per Sample')
        axes[0,1].set_title('Security Degradation with LO Noise')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3, which='both')
        
        # 3. Comparison with paper
        paper_rate = 580.7
        bbo_rates = []
        ppktp_rates = []
        
        for sq in squeezing_levels:
            # BBO with realistic noise
            res_bbo = self.simulate_complete_protocol(sq, 0.05, -13)
            bbo_rates.append(res_bbo['rate_mbps'])
            
            # PPKTP with lower noise (from paper)
            res_ppktp = self.simulate_complete_protocol(sq, 0.01, -15)
            ppktp_rates.append(res_ppktp['rate_mbps'])
        
        axes[1,0].plot(squeezing_levels, bbo_rates, 'b-', label='BBO (Our Target)', linewidth=2)
        axes[1,0].plot(squeezing_levels, ppktp_rates, 'r-', label='PPKTP (Paper)', linewidth=2)
        axes[1,0].axhline(y=paper_rate, color='g', linestyle='--', label=f'Paper: {paper_rate} Mbps')
        axes[1,0].set_xlabel('Squeezing Level (dB)')
        axes[1,0].set_ylabel('Secure Rate (Mbps)')
        axes[1,0].set_title('BBO vs PPKTP Performance Comparison')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Optimal operating point
        LO_range = np.linspace(0.01, 0.2, 20)
        squeeze_range = np.linspace(2, 6, 20)
        
        X, Y = np.meshgrid(LO_range, squeeze_range)
        Z = np.zeros_like(X)
        
        for i in tqdm(range(len(squeeze_range)), desc="Optimization grid"):
            for j in range(len(LO_range)):
                res = self.simulate_complete_protocol(
                    squeeze_range[i], LO_range[j], -13)
                Z[i, j] = res['rate_mbps']
        
        contour = axes[1,1].contourf(X, Y, Z, 20, cmap='viridis')
        axes[1,1].set_xlabel('LO Noise Fraction')
        axes[1,1].set_ylabel('Squeezing Level (dB)')
        axes[1,1].set_title('Optimal Operating Region')
        plt.colorbar(contour, ax=axes[1,1], label='Rate (Mbps)')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('security_simulation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig