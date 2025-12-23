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
        CORRECTED VERSION: Proper entropy calculation matching paper
        
        Based on paper Eq. 6: H_min(P|E) ≥ -log2(c(δq,δp)) - H_max(Q)
        With proper normalization and realistic values
        """
        # DEBUG
        print(f"\n[ENTROPY_CALC] Input: squeezing={squeezing_db}dB, LO_noise={LO_noise_fraction}, elec={electronic_noise_db}dB")
        
        # 1. Convert to linear scale (shot noise = 1)
        squeezing_linear = 10**(squeezing_db/10)  # > 1 for anti-squeezed
        
        # Squeezed quadrature variance (below shot noise)
        var_squeezed = 1 / squeezing_linear  # < 1
        
        # Anti-squeezed quadrature variance (above shot noise)
        var_anti = squeezing_linear  # > 1
        
        print(f"[ENTROPY_CALC] Linear: {squeezing_linear:.3f}, var_squeezed={var_squeezed:.3f}, var_anti={var_anti:.3f}")
        
        # 2. Add measurement noises (all normalized to shot noise = 1)
        electronic_noise = 10**(electronic_noise_db/10)
        
        # Total variances including noise
        var_squeezed_total = var_squeezed + LO_noise_fraction + electronic_noise
        var_anti_total = var_anti + LO_noise_fraction + electronic_noise
        
        print(f"[ENTROPY_CALC] With noise: var_squeezed_total={var_squeezed_total:.3f}, var_anti_total={var_anti_total:.3f}")
        
        # 3. Calculate H_max(Q) for squeezed quadrature
        # For Gaussian distribution: H_max = log2(√(2πeσ²)/δ)
        # From paper: H_max = log2(measurement range / resolution)
        # Simpler: H_max ≈ log2(k * σ/δ) where k depends on distribution
        
        # Using Gaussian approximation from paper's Supplementary
        delta = self.params['delta']
        H_max = 0.5 * np.log2(2 * np.pi * np.e * var_squeezed_total / (delta**2))
        
        print(f"[ENTROPY_CALC] H_max = {H_max:.4f}")
        
        # 4. Incompatibility term c(δq,δp)
        # From paper: c(δq,δp) = δq·δp/(2π) * S₀¹(1, δq·δp/4)²
        # For small δ: c ≈ δ²/(2π)
        c_term = delta**2 / (2 * np.pi)
        print(f"[ENTROPY_CALC] c_term = {c_term:.6f}, -log2(c) = {-np.log2(c_term):.4f}")
        
        # 5. Lower bound on conditional min-entropy (Eq. 6)
        H_min_bound = -np.log2(c_term) - H_max
        print(f"[ENTROPY_CALC] H_min_bound (basic) = {H_min_bound:.4f}")
        
        # 6. Account for noise - only quantum part is secure!
        # Quantum fraction = quantum variance / total variance
        quantum_fraction = var_anti / var_anti_total
        print(f"[ENTROPY_CALC] quantum_fraction = {quantum_fraction:.4f}")
        
        # Trusted fraction = 1 - (untrusted noise / total noise)
        untrusted_noise = LO_noise_fraction + electronic_noise
        trusted_fraction = 1 - (untrusted_noise / var_anti_total)
        print(f"[ENTROPY_CALC] trusted_fraction = {trusted_fraction:.4f}")
        
        
        H_min_final = H_min_bound * quantum_fraction * trusted_fraction
        print(f"[ENTROPY_CALC] H_min_final = {H_min_final:.4f}")
        
        return H_min_final, H_max
    
    def simulate_complete_protocol(self, squeezing_db, LO_noise_fraction, 
                                electronic_noise_db, N_measurements=1e6):
        """
        CORRECTED: Simulate complete SDI-QRNG protocol matching paper
        """
        print(f"\n[PROTOCOL] squeezing={squeezing_db}dB, LO={LO_noise_fraction}, elec={electronic_noise_db}dB")
        
        # 1. Calculate entropy bounds
        H_min, H_max = self.entropic_uncertainty_bound(
            squeezing_db, LO_noise_fraction, electronic_noise_db)
        
        print(f"[PROTOCOL] Raw: H_min={H_min:.4f}, H_max={H_max:.4f}")
        
        # 2. Apply finite-size effects (Eq. 7 from paper)
        delta_term = 4 * np.sqrt(np.log2(2/self.params['epsilon']**2))
        delta_term *= np.log2(2**(1 + H_max/2) + 1)
        
        H_min_smooth = H_min - delta_term / np.sqrt(N_measurements)
        H_min_smooth = max(0, H_min_smooth)  # Ensure non-negative
        
        print(f"[PROTOCOL] After finite-size: H_min_smooth={H_min_smooth:.4f}")
        
        # 3. Account for ADC resolution - FROM PAPER'S NUMBERS!
        # Paper: H_min=7.21 → after 6-bit ADC → 3.32 bits
        # So ADC efficiency = 3.32/7.21 = 0.46
        
        # If entropy > what ADC can resolve, we lose information
        # Paper uses: effective_bits = min(H_min_smooth, ADC_bits) * ADC_efficiency
        adc_bits = self.params['oscillator_bits']
        adc_efficiency = 3.32 / 7.21  # From paper
        
        secure_bits_before_hash = H_min_smooth * adc_efficiency

        # But practical limit: can't extract more than ~6 bits realistically
        max_realistic_bits = 6.0
        secure_bits_before_hash = min(secure_bits_before_hash, max_realistic_bits)

        
        print(f"[PROTOCOL] ADC: bits={adc_bits}, eff={adc_efficiency:.3f}, before_hash={secure_bits_before_hash:.4f}")
        
        # 4. Apply hashing efficiency
        secure_bits_per_sample = secure_bits_before_hash * self.params['hashing_efficiency']
        
        print(f"[PROTOCOL] After hashing: {secure_bits_per_sample:.4f} bits/sample")
        
        # 5. Calculate final rate
        rate_bps = secure_bits_per_sample * self.params['sampling_rate']
        rate_mbps = rate_bps / 1e6
        
        print(f"[PROTOCOL] Final rate: {rate_mbps:.2f} Mbps")
        
        results = {
            'H_min_bound': H_min,
            'H_min_smooth': H_min_smooth,
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
    

