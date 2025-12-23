"""
Homodyne Detection Simulation
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
import qutip as qt

class HomodyneDetector:
    """
    Simulate balanced homodyne detection
    """
    
    def __init__(self, sampling_rate=1e9, N_samples=100000):
        self.fs = sampling_rate
        self.N = N_samples
        self.time = np.arange(N_samples) / sampling_rate
        
        # Detector parameters
        self.params = {
            'quantum_efficiency': 0.88,
            'electronic_noise': -13,  # dB below SNL
            'bandwidth': 200e6,       # Hz
            'dark_current': 10e-9,    # A
            'responsivity': 0.8       # A/W
        }
    
    def generate_quantum_state(self, squeezing_db, phase=0):
        """
        Generate squeezed state using QuTiP
        """
        # Squeezing parameter
        r = squeezing_db * np.log(10) / (20 * np.log(np.exp(1)))
        
        # Create squeezed vacuum state
        N_cutoff = 20  # Fock state cutoff
        squeezed_state = qt.squeeze(N_cutoff, r * np.exp(1j * phase))
        
        # Calculate quadrature variances
        a = qt.destroy(N_cutoff)
        X = (a + a.dag()) / np.sqrt(2)  # Amplitude quadrature
        P = (a - a.dag()) / (1j * np.sqrt(2))  # Phase quadrature
        
        var_X = qt.expect(X**2, squeezed_state)
        var_P = qt.expect(P**2, squeezed_state)
        
        return squeezed_state, var_X, var_P
    
    def simulate_detection(self, squeezing_db, LO_power, phase_scan=False):
        """
        Simulate homodyne detection
        """
        # Generate time-domain signals
        t = self.time
        
        # Quantum noise (shot noise limited)
        shot_noise = np.random.normal(0, 1, self.N)
        
        # Apply squeezing transformation
        r = squeezing_db * np.log(10) / (20 * np.log(np.exp(1)))
        
        if phase_scan:
            # Scan phase from 0 to 2π
            phases = np.linspace(0, 2*np.pi, 361)
            variances = []
            
            for phi in phases:
                # Quadrature operator
                X_phi = np.cos(phi) * shot_noise * np.exp(-r) + \
                       np.sin(phi) * shot_noise * np.exp(r)
                
                # Add LO signal
                LO_signal = np.sqrt(LO_power) * np.cos(phi)
                detector_signal = LO_signal + X_phi
                
                variances.append(np.var(detector_signal))
            
            return phases, variances
        
        else:
            # Fixed phase measurement
            # Anti-squeezed quadrature (for randomness)
            X_anti = shot_noise * np.exp(r)
            
            # Add LO
            LO_amplitude = np.sqrt(LO_power)
            detector_output = LO_amplitude + X_anti
            
            # Add electronic noise
            elec_noise_level = 10**(self.params['electronic_noise']/10)
            elec_noise = np.random.normal(0, np.sqrt(elec_noise_level), self.N)
            
            detector_output += elec_noise
            
            # Apply bandwidth filter
            b, a = signal.butter(4, self.params['bandwidth']/(self.fs/2), 'low')
            filtered_output = signal.filtfilt(b, a, detector_output)
            
            return t, filtered_output
    
    def calculate_SNR(self, squeezing_db, LO_power):
        """
        Calculate Signal-to-Noise Ratio
        """
        # Signal power (LO contribution)
        signal_power = LO_power
        
        # Noise powers
        shot_noise_power = 1  # Normalized
        squeezing_factor = 10**(squeezing_db/10)
        
        if squeezing_db < 0:
            # Squeezed quadrature
            quantum_noise = shot_noise_power * squeezing_factor
        else:
            # Anti-squeezed quadrature
            quantum_noise = shot_noise_power * squeezing_factor
        
        electronic_noise = 10**(self.params['electronic_noise']/10)
        
        total_noise = quantum_noise + electronic_noise
        
        SNR = signal_power / total_noise
        
        return SNR, {
            'quantum_noise': quantum_noise,
            'electronic_noise': electronic_noise,
            'total_noise': total_noise
        }
    
    def plot_detection_results(self, squeezing_db=3.0, LO_power=10):
        """
        Plot comprehensive detection results
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Phase scan
        phases, variances = self.simulate_detection(squeezing_db, LO_power, phase_scan=True)
        axes[0,0].plot(phases, 10*np.log10(variances), 'b-', linewidth=2)
        axes[0,0].axhline(y=0, color='r', linestyle='--', label='Shot Noise Limit')
        axes[0,0].set_xlabel('Measurement Phase (rad)')
        axes[0,0].set_ylabel('Noise Power (dB)')
        axes[0,0].set_title('Phase-Dependent Noise Measurement')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Time-domain signals
        t, signal = self.simulate_detection(squeezing_db, LO_power, phase_scan=False)
        axes[0,1].plot(t[:1000]*1e9, signal[:1000], 'b-', alpha=0.7)
        axes[0,1].set_xlabel('Time (ns)')
        axes[0,1].set_ylabel('Detector Output (a.u.)')
        axes[0,1].set_title('Time-Domain Signal (First 1000 samples)')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Histogram
        axes[1,0].hist(signal, bins=100, density=True, alpha=0.7, 
                      color='blue', edgecolor='black')
        x = np.linspace(np.min(signal), np.max(signal), 1000)
        gaussian_fit = stats.norm.pdf(x, np.mean(signal), np.std(signal))
        axes[1,0].plot(x, gaussian_fit, 'r-', linewidth=2, label='Gaussian Fit')
        axes[1,0].set_xlabel('Signal Amplitude')
        axes[1,0].set_ylabel('Probability Density')
        axes[1,0].set_title('Signal Distribution')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. SNR vs LO power
        LO_powers = np.logspace(-1, 2, 50)  # 0.1 to 100 mW
        SNRs = []
        
        for P in LO_powers:
            SNR, _ = self.calculate_SNR(squeezing_db, P)
            SNRs.append(10*np.log10(SNR))
        
        axes[1,1].plot(10*np.log10(LO_powers), SNRs, 'g-', linewidth=2)
        axes[1,1].set_xlabel('LO Power (dB)')
        axes[1,1].set_ylabel('SNR (dB)')
        axes[1,1].set_title('Signal-to-Noise Ratio vs LO Power')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('detection_simulation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig