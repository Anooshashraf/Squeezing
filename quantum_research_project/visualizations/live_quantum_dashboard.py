"""
Live Quantum Dashboard with Real-time Simulation
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
from datetime import datetime
import time

class QuantumDashboard:
    """Live dashboard showing all quantum processes"""
    
    def __init__(self):
        self.fig = plt.figure(figsize=(20, 12))
        self.gs = GridSpec(3, 4, figure=self.fig)
        
        # Initialize subplots
        self.setup_dashboard()
        
        # Simulation parameters
        self.time = 0
        self.data_buffer = []
        self.bit_buffer = []
        
    def setup_dashboard(self):
        """Setup all dashboard panels"""
        
        # 1. Optical Setup Schematic (top left)
        self.ax_schematic = self.fig.add_subplot(self.gs[0, 0])
        self.ax_schematic.set_title('🔧 Optical Setup Schematic', 
                                   fontsize=14, fontweight='bold')
        self.ax_schematic.axis('equal')
        self.ax_schematic.set_xlim(0, 10)
        self.ax_schematic.set_ylim(0, 10)
        self.ax_schematic.axis('off')
        
        # 2. Quantum Squeezing Visualization (top middle)
        self.ax_squeeze = self.fig.add_subplot(self.gs[0, 1])
        self.ax_squeeze.set_title('💎 Quantum Squeezing Visualization', 
                                 fontsize=14, fontweight='bold')
        
        # 3. Homodyne Detection Signals (top right)
        self.ax_signals = self.fig.add_subplot(self.gs[0, 2])
        self.ax_signals.set_title('📊 Homodyne Detection Signals', 
                                 fontsize=14, fontweight='bold')
        self.ax_signals.set_xlabel('Time (ns)')
        self.ax_signals.set_ylabel('Amplitude')
        self.ax_signals.grid(True, alpha=0.3)
        
        # 4. Phase Space Diagram (middle left)
        self.ax_phase = self.fig.add_subplot(self.gs[1, 0])
        self.ax_phase.set_title('🌀 Phase Space Representation', 
                               fontsize=14, fontweight='bold')
        self.ax_phase.set_xlabel('X-quadrature')
        self.ax_phase.set_ylabel('P-quadrature')
        self.ax_phase.grid(True, alpha=0.3)
        
        # 5. Noise Power Spectrum (middle)
        self.ax_spectrum = self.fig.add_subplot(self.gs[1, 1])
        self.ax_spectrum.set_title('📈 Noise Power Spectrum', 
                                  fontsize=14, fontweight='bold')
        self.ax_spectrum.set_xlabel('Frequency (MHz)')
        self.ax_spectrum.set_ylabel('Power (dB)')
        self.ax_spectrum.grid(True, alpha=0.3)
        
        # 6. Bit Generation Monitor (middle right)
        self.ax_bits = self.fig.add_subplot(self.gs[1, 2])
        self.ax_bits.set_title('🎲 Random Bit Generation', 
                              fontsize=14, fontweight='bold')
        self.ax_bits.set_xlabel('Bit Position')
        self.ax_bits.set_ylabel('Bit Value')
        self.ax_bits.grid(True, alpha=0.3)
        
        # 7. System Parameters (bottom left)
        self.ax_params = self.fig.add_subplot(self.gs[2, 0])
        self.ax_params.set_title('⚙️ System Parameters', 
                                fontsize=14, fontweight='bold')
        self.ax_params.axis('off')
        
        # 8. Performance Metrics (bottom middle)
        self.ax_metrics = self.fig.add_subplot(self.gs[2, 1])
        self.ax_metrics.set_title('📊 Performance Metrics', 
                                 fontsize=14, fontweight='bold')
        self.ax_metrics.axis('off')
        
        # 9. Security Analysis (bottom right)
        self.ax_security = self.fig.add_subplot(self.gs[2, 2])
        self.ax_security.set_title('🔒 Security Analysis', 
                                  fontsize=14, fontweight='bold')
        self.ax_security.axis('off')
        
        # 10. Log/Status Panel (right column)
        self.ax_log = self.fig.add_subplot(self.gs[:, 3])
        self.ax_log.set_title('📝 System Log & Status', 
                             fontsize=14, fontweight='bold')
        self.ax_log.axis('off')
        
    def draw_schematic(self):
        """Draw optical setup schematic"""
        self.ax_schematic.clear()
        self.ax_schematic.set_xlim(0, 10)
        self.ax_schematic.set_ylim(0, 10)
        self.ax_schematic.axis('off')
        
        # Components
        components = [
            (1, 5, '🔦', 'Laser\n532 nm', '#FF6B6B'),
            (3, 5, '🔍', 'Beam\nShaper', '#4ECDC4'),
            (5, 5, '🔥💎', 'BBO\nCrystal', '#06D6A0'),
            (7, 6, '💡', 'Local\nOscillator', '#118AB2'),
            (7, 4, '↔', 'Beam\nSplitter', '#8AC926'),
            (8.5, 6.5, '📊', 'Detector\nA', '#FF595E'),
            (8.5, 3.5, '📊', 'Detector\nB', '#FF595E'),
            (9.5, 5, '➖', 'Balanced\nDetection', '#FFD166'),
            (10.5, 5, '📈', 'Analysis', '#6A4C93'),
        ]
        
        for x, y, emoji, label, color in components:
            # Draw component box
            box = plt.Rectangle((x-0.4, y-0.4), 0.8, 0.8,
                               facecolor=color, alpha=0.3,
                               edgecolor='black', linewidth=1)
            self.ax_schematic.add_patch(box)
            
            # Add emoji
            self.ax_schematic.text(x, y+0.1, emoji, fontsize=16,
                                  ha='center', va='center')
            
            # Add label
            self.ax_schematic.text(x, y-0.6, label, fontsize=8,
                                  ha='center', va='center')
        
        # Draw light paths
        paths = [
            [(1.4, 5), (2.6, 5)],  # Laser to beam shaper
            [(3.4, 5), (4.6, 5)],  # Beam shaper to BBO
            [(5.4, 5), (6.6, 4)],  # BBO to beam splitter
            [(7, 5.6), (7, 5.4)],  # LO to BS vertical
            [(7.4, 4), (8.1, 3.5)], # BS to detector B
            [(7.4, 4), (8.1, 6.5)], # BS to detector A
            [(8.9, 3.5), (9.1, 4.5)], # Det B to subtractor
            [(8.9, 6.5), (9.1, 5.5)], # Det A to subtractor
            [(9.9, 5), (10.1, 5)], # Subtractor to analysis
        ]
        
        for (x1, y1), (x2, y2) in paths:
            self.ax_schematic.arrow(x1, y1, x2-x1, y2-y1,
                                   head_width=0.1, head_length=0.1,
                                   fc='red', ec='red', alpha=0.6)
        
        # Add animated photons
        t = self.time
        for i in range(5):
            x_photon = 1.5 + i*0.5 + 0.2*np.sin(t + i)
            y_photon = 5 + 0.1*np.sin(2*t + i)
            photon = plt.Circle((x_photon, y_photon), 0.05,
                               color='#00FF00', alpha=0.7)
            self.ax_schematic.add_patch(photon)
    
    def draw_squeezing(self):
        """Draw squeezing visualization"""
        self.ax_squeeze.clear()
        
        t = self.time
        
        # Create squeezing visualization
        angles = np.linspace(0, 2*np.pi, 100)
        
        # Shot noise circle
        circle_x = np.cos(angles)
        circle_y = np.sin(angles)
        self.ax_squeeze.plot(circle_x, circle_y, '--', 
                            color='gray', alpha=0.5, label='Shot Noise')
        
        # Squeezed ellipse
        squeezing = 3.0 + 1.0 * np.sin(t * 0.5)  # 2-4 dB varying
        
        # Anti-squeezing (noise amplification)
        anti_squeeze = 1.0 / (10**(-squeezing/20))
        
        ellipse_x = np.cos(angles) * anti_squeeze
        ellipse_y = np.sin(angles) / anti_squeeze
        
        # Rotate based on measurement phase
        phase = t * 2
        rot_x = ellipse_x * np.cos(phase) - ellipse_y * np.sin(phase)
        rot_y = ellipse_x * np.sin(phase) + ellipse_y * np.cos(phase)
        
        self.ax_squeeze.fill(rot_x, rot_y, color='blue', alpha=0.2)
        self.ax_squeeze.plot(rot_x, rot_y, '-', color='darkblue', 
                            linewidth=2, label=f'Squeezed ({squeezing:.1f} dB)')
        
        # Mark quadratures
        self.ax_squeeze.arrow(0, 0, 1.5, 0, head_width=0.05, 
                             head_length=0.1, fc='black', ec='black')
        self.ax_squeeze.arrow(0, 0, 0, 1.5, head_width=0.05, 
                             head_width=0.05, fc='black', ec='black')
        
        self.ax_squeeze.text(1.6, 0, 'X (Amplitude)', fontsize=10)
        self.ax_squeeze.text(0, 1.6, 'P (Phase)', fontsize=10)
        
        # Highlight anti-squeezed direction
        if anti_squeeze > 1:
            angle = phase
            x_marker = 1.8 * np.cos(angle)
            y_marker = 1.8 * np.sin(angle)
            self.ax_squeeze.text(x_marker, y_marker, '⭐', 
                                fontsize=20, ha='center', va='center',
                                color='gold')
            self.ax_squeeze.text(x_marker*1.2, y_marker*1.2, 
                                'Randomness\nSource', fontsize=8,
                                ha='center', color='darkblue')
        
        self.ax_squeeze.set_xlim(-2, 2)
        self.ax_squeeze.set_ylim(-2, 2)
        self.ax_squeeze.set_aspect('equal')
        self.ax_squeeze.legend(loc='upper right')
        self.ax_squeeze.grid(True, alpha=0.3)
    
    def draw_signals(self):
        """Draw homodyne detection signals"""
        self.ax_signals.clear()
        
        t = np.linspace(0, 20, 200)  # 20 ns window
        
        # Generate signals
        signal_A = 0.5 * np.sin(2*np.pi*0.5*t + self.time)  # Carrier
        signal_A += 0.3 * np.random.randn(len(t))  # Quantum noise
        signal_A += 0.1 * np.sin(2*np.pi*0.1*t)  # Low freq noise
        
        signal_B = 0.5 * np.sin(2*np.pi*0.5*t + self.time + np.pi)  # Anti-phase
        signal_B += 0.3 * np.random.randn(len(t))  # Quantum noise
        signal_B += 0.1 * np.sin(2*np.pi*0.1*t + np.pi/2)  # Low freq noise
        
        # Difference signal (quantum noise)
        diff_signal = signal_B - signal_A
        
        # Plot signals
        self.ax_signals.plot(t, signal_A, '-', color='red', 
                            alpha=0.7, label='Detector A', linewidth=1)
        self.ax_signals.plot(t, signal_B, '-', color='blue', 
                            alpha=0.7, label='Detector B', linewidth=1)
        self.ax_signals.plot(t, diff_signal, '-', color='green', 
                            alpha=0.9, label='Difference (Quantum Noise)', 
                            linewidth=2)
        
        self.ax_signals.set_xlabel('Time (ns)')
        self.ax_signals.set_ylabel('Amplitude')
        self.ax_signals.legend(loc='upper right')
        self.ax_signals.grid(True, alpha=0.3)
        
        # Store for statistics
        self.data_buffer.extend(diff_signal.tolist())
        if len(self.data_buffer) > 1000:
            self.data_buffer = self.data_buffer[-1000:]
    
    def draw_phase_space(self):
        """Draw phase space diagram"""
        self.ax_phase.clear()
        
        if len(self.data_buffer) > 100:
            # Use recent data for phase space
            data = np.array(self.data_buffer[-100:])
            
            # Create quadrature pairs (X and P)
            X = data[:-1]
            P = data[1:]
            
            # Scatter plot
            self.ax_phase.scatter(X, P, c='blue', alpha=0.6, s=10,
                                 edgecolors='none')
            
            # Add uncertainty ellipse
            from matplotlib.patches import Ellipse
            cov = np.cov(X, P)
            lambda_, v = np.linalg.eig(cov)
            lambda_ = np.sqrt(lambda_)
            
            ellipse = Ellipse(xy=(np.mean(X), np.mean(P)),
                             width=lambda_[0]*2, height=lambda_[1]*2,
                             angle=np.degrees(np.arctan2(v[1,0], v[0,0])),
                             edgecolor='red', facecolor='none',
                             linewidth=2, linestyle='--')
            self.ax_phase.add_patch(ellipse)
        
        self.ax_phase.set_xlabel('X-quadrature')
        self.ax_phase.set_ylabel('P-quadrature')
        self.ax_phase.grid(True, alpha=0.3)
        self.ax_phase.set_title('Phase Space Distribution')
    
    def draw_spectrum(self):
        """Draw noise power spectrum"""
        self.ax_spectrum.clear()
        
        if len(self.data_buffer) > 100:
            # Calculate FFT
            data = np.array(self.data_buffer[-512:])
            spectrum = np.abs(np.fft.fft(data))**2
            freqs = np.fft.fftfreq(len(data), d=1e-9) / 1e6  # MHz
            
            # Only positive frequencies
            pos_mask = freqs >= 0
            freqs = freqs[pos_mask]
            spectrum = spectrum[pos_mask]
            
            self.ax_spectrum.plot(freqs, 10*np.log10(spectrum), 
                                 '-', color='purple', linewidth=2)
            
            # Mark regions
            self.ax_spectrum.axvline(x=3, color='red', linestyle='--',
                                    alpha=0.5, label='3 MHz cutoff')
            self.ax_spectrum.axvline(x=200, color='red', linestyle='--',
                                    alpha=0.5, label='200 MHz bandwidth')
            self.ax_spectrum.fill_betweenx([-100, 50], 3, 200,
                                          color='green', alpha=0.1,
                                          label='QRNG Bandwidth')
            
            # Shot noise reference
            self.ax_spectrum.axhline(y=0, color='gray', linestyle='--',
                                    alpha=0.5, label='Shot Noise Level')
        
        self.ax_spectrum.set_xlabel('Frequency (MHz)')
        self.ax_spectrum.set_ylabel('Power (dB)')
        self.ax_spectrum.set_xlim(0, 250)
        self.ax_spectrum.set_ylim(-30, 30)
        self.ax_spectrum.legend(loc='upper right', fontsize=8)
        self.ax_spectrum.grid(True, alpha=0.3)
    
    def draw_bits(self):
        """Draw random bit generation"""
        self.ax_bits.clear()
        
        # Generate new bits occasionally
        if np.random.rand() < 0.1:  # 10% chance each frame
            new_bits = np.random.randint(0, 2, 8)
            self.bit_buffer.extend(new_bits.tolist())
        
        if len(self.bit_buffer) > 64:
            self.bit_buffer = self.bit_buffer[-64:]
        
        if len(self.bit_buffer) > 0:
            # Display as bar chart
            positions = np.arange(len(self.bit_buffer))
            colors = ['green' if b == 1 else 'red' for b in self.bit_buffer]
            
            self.ax_bits.bar(positions, self.bit_buffer, 
                            color=colors, edgecolor='black')
            
            # Add bit values on top
            for i, bit in enumerate(self.bit_buffer):
                self.ax_bits.text(i, bit + 0.05, str(bit),
                                 ha='center', va='bottom', fontsize=8)
        
        self.ax_bits.set_xlabel('Bit Position')
        self.ax_bits.set_ylabel('Bit Value (0/1)')
        self.ax_bits.set_ylim(-0.1, 1.6)
        self.ax_bits.grid(True, alpha=0.3, axis='y')
    
    def draw_parameters(self):
        """Draw system parameters"""
        self.ax_params.clear()
        self.ax_params.axis('off')
        
        t = self.time
        
        params = [
            ('⚙️ SYSTEM PARAMETERS', ''),
            ('Laser Power:', f'{200 + 10*np.sin(t):.0f} mW'),
            ('BBO Temperature:', f'{25 + 0.5*np.sin(t*0.5):.1f} °C'),
            ('Squeezing Level:', f'{3 + np.sin(t*0.3):.1f} dB'),
            ('Detection Bandwidth:', '3-200 MHz'),
            ('Sampling Rate:', '200 MSamples/s'),
            ('ADC Resolution:', '8 bits'),
            ('LO Power:', '10 mW'),
            ('Visibility:', '99%'),
            ('Quantum Efficiency:', '88%'),
        ]
        
        y_pos = 0.9
        for label, value in params:
            if 'SYSTEM' in label:
                self.ax_params.text(0.1, y_pos, label, 
                                   fontsize=12, fontweight='bold',
                                   color='darkblue')
            else:
                self.ax_params.text(0.1, y_pos, f'{label:20}', 
                                   fontsize=10, ha='left')
                self.ax_params.text(0.7, y_pos, value, 
                                   fontsize=10, ha='right',
                                   fontweight='bold')
            y_pos -= 0.07
    
    def draw_metrics(self):
        """Draw performance metrics"""
        self.ax_metrics.clear()
        self.ax_metrics.axis('off')
        
        t = self.time
        
        metrics = [
            ('📊 PERFORMANCE METRICS', ''),
            ('Bit Generation Rate:', f'{250 + 50*np.sin(t*0.2):.0f} Mbps'),
            ('Min-Entropy:', f'{7.2 + 0.2*np.sin(t):.1f} bits/sample'),
            ('SNR:', f'{15 + 2*np.sin(t*0.5):.1f} dB'),
            ('Excess Noise:', f'{0.05 + 0.01*np.sin(t):.3f}'),
            ('Hashing Efficiency:', '95%'),
            ('Secure Bits/Sample:', f'{2.5 + 0.5*np.sin(t*0.3):.2f}'),
            ('System Uptime:', f'{t/10:.1f} s'),
            ('Bits Generated:', f'{len(self.bit_buffer)}'),
            ('Data Processed:', f'{len(self.data_buffer)/1000:.1f} kSamples'),
        ]
        
        y_pos = 0.9
        for label, value in metrics:
            if 'PERFORMANCE' in label:
                self.ax_metrics.text(0.1, y_pos, label, 
                                    fontsize=12, fontweight='bold',
                                    color='darkgreen')
            else:
                self.ax_metrics.text(0.1, y_pos, f'{label:20}', 
                                    fontsize=10, ha='left')
                self.ax_metrics.text(0.7, y_pos, value, 
                                    fontsize=10, ha='right',
                                    fontweight='bold',
                                    color='green' if 'Mbps' in value else 'black')
            y_pos -= 0.07
    
    def draw_security(self):
        """Draw security analysis"""
        self.ax_security.clear()
        self.ax_security.axis('off')
        
        t = self.time
        
        security_info = [
            ('🔒 SECURITY ANALYSIS', ''),
            ('Protocol:', 'Semi-Device-Independent'),
            ('Trusted:', 'Detector & Processor'),
            ('Untrusted:', 'Source & Local Oscillator'),
            ('Security Parameter (ε):', '10⁻⁶'),
            ('Conditional Min-Entropy:', f'{7.0 + 0.5*np.sin(t):.2f}'),
            ('LO Noise Fraction:', f'{0.05 + 0.01*np.sin(t*0.7):.3f}'),
            ('Electronic Noise:', '-13 dB'),
            ('Attack Detection:', 'Active'),
            ('Status:', 'SECURE ✅'),
        ]
        
        y_pos = 0.9
        for label, value in security_info:
            if 'SECURITY' in label:
                self.ax_security.text(0.1, y_pos, label, 
                                     fontsize=12, fontweight='bold',
                                     color='darkred')
            else:
                self.ax_security.text(0.1, y_pos, f'{label:25}', 
                                     fontsize=10, ha='left')
                color = 'green' if 'SECURE' in value else \
                       'red' if 'ALERT' in value else 'black'
                self.ax_security.text(0.8, y_pos, value, 
                                     fontsize=10, ha='right',
                                     fontweight='bold', color=color)
            y_pos -= 0.07
    
    def draw_log(self):
        """Draw system log"""
        self.ax_log.clear()
        self.ax_log.axis('off')
        
        # Simulated log entries
        log_entries = [
            f"[{datetime.now().strftime('%H:%M:%S')}] System initialized",
            f"[{datetime.now().strftime('%H:%M:%S')}] Laser ON - 532 nm, 200 mW",
            f"[{datetime.now().strftime('%H:%M:%S')}] BBO temperature stabilized",
            f"[{datetime.now().strftime('%H:%M:%S')}] Squeezing detected: {3.2 + 0.1*np.sin(self.time):.1f} dB",
            f"[{datetime.now().strftime('%H:%M:%S')}] Homodyne detection active",
            f"[{datetime.now().strftime('%H:%M:%S')}] Quantum noise variance: {0.95 + 0.1*np.sin(self.time*0.5):.3f}",
            f"[{datetime.now().strftime('%H:%M:%S')}] Generating random bits at {250 + 50*np.sin(self.time*0.2):.0f} Mbps",
            f"[{datetime.now().strftime('%H:%M:%S')}] Security check passed",
            f"[{datetime.now().strftime('%H:%M:%S')}] NIST tests: All passing",
            f"[{datetime.now().strftime('%H:%M:%S')}] Total bits generated: {len(self.bit_buffer)}",
        ]
        
        # Display log
        y_pos = 0.95
        for entry in log_entries[-10:]:  # Last 10 entries
            color = 'green' if 'passed' in entry.lower() else \
                   'red' if 'error' in entry.lower() else \
                   'blue' if 'initialized' in entry else \
                   'black'
            
            self.ax_log.text(0.02, y_pos, entry, fontsize=9,
                            ha='left', va='top', color=color)
            y_pos -= 0.08
        
        # Add status indicator
        status_color = 'green' if np.sin(self.time) > -0.5 else 'red'
        status_text = 'NORMAL OPERATION' if status_color == 'green' else 'CHECK SYSTEM'
        
        self.ax_log.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.1,
                                           facecolor=status_color, alpha=0.2,
                                           edgecolor=status_color, linewidth=2))
        self.ax_log.text(0.5, 0.07, f'🚀 STATUS: {status_text}', 
                        fontsize=12, ha='center', fontweight='bold',
                        color=status_color)
    
    def update(self, frame):
        """Update all dashboard panels"""
        self.time = frame * 0.1  # Increment time
        
        # Update all panels
        self.draw_schematic()
        self.draw_squeezing()
        self.draw_signals()
        self.draw_phase_space()
        self.draw_spectrum()
        self.draw_bits()
        self.draw_parameters()
        self.draw_metrics()
        self.draw_security()
        self.draw_log()
        
        # Update title with timestamp
        self.fig.suptitle(f'🌌 BBO SQUEEZED LIGHT QRNG - LIVE DASHBOARD | Time: {self.time:.1f}s', 
                         fontsize=16, fontweight='bold', color='darkblue')
        
        return []
    
    def run(self):
        """Run the dashboard animation"""
        print("🚀 Starting Quantum Dashboard...")
        print("Dashboard shows:")
        print("1. Optical schematic with animated photons")
        print("2. Quantum squeezing visualization")
        print("3. Real-time detection signals")
        print("4. Phase space distribution")
        print("5. Noise spectrum")
        print("6. Random bit generation")
        print("7. System parameters")
        print("8. Performance metrics")
        print("9. Security analysis")
        print("10. System log")
        
        anim = animation.FuncAnimation(
            self.fig, 
            self.update,
            frames=1000,  # Run for 1000 frames
            interval=100,  # Update every 100ms (10 fps)
            blit=False,
            repeat=True
        )
        
        plt.tight_layout()
        plt.show()
        
        return anim

# Run the dashboard
if __name__ == "__main__":
    dashboard = QuantumDashboard()
    anim = dashboard.run()