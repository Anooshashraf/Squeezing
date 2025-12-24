"""
Interactive Animation of BBO Squeezed Light QRNG
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, Polygon, FancyArrowPatch
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as path_effects

class QuantumAnimation:
    """animation of the complete QRNG process"""
    
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(16, 10))
        self.ax.set_xlim(-1, 15)
        self.ax.set_ylim(-1, 9)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        # Animation parameters
        self.frame = 0
        self.max_frames = 360
        
        # Store animation objects
        self.laser_beam = []
        self.squeezed_particles = []
        self.detector_signals = []
        
        # Initialize components
        self.setup_components()
        
    def setup_components(self):
        """Setup all optical components"""
        
        # Title (move up slightly)
        self.ax.text(7, 8.7, 'BBO SQUEEZED LIGHT QRNG - LIVE SIMULATION', 
                    fontsize=18, fontweight='bold', ha='center',
                    color='darkblue',
                    path_effects=[path_effects.withStroke(linewidth=3, 
                                                         foreground='white')])
        # Instructions box (move below title, left-aligned)
        explanation = """
        HOW IT WORKS:
        1. Green laser (532 nm) pumps BBO crystal
        2. BBO in temperature-controlled oven creates nonlinear effects
        3. Quantum squeezing: Blue = Anti-squeezed (more noise = randomness)
        4. ↔Beam splitter combines with reference laser
        5. Two detectors measure interference
        6. Subtractor finds difference (quantum noise)
        7. Oscilloscope analyzes quantum fluctuations
        8. Quantum noise → True random bits!
        """
        self.ax.text(1.2, 7.8, explanation, fontsize=10, 
                    ha='left', va='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5, edgecolor='navy'))
        
        # 1. LASER SOURCE
        self.ax.add_patch(Rectangle((0.5, 3.5), 1, 1, 
                                   facecolor='#FF6B6B', edgecolor='black', 
                                   linewidth=2))
        self.ax.text(1, 4, 'Laser', fontsize=20, ha='center', va='center')
        self.ax.text(1, 3, '532 nm\nGreen Laser', fontsize=10, ha='center', 
                    fontweight='bold')
        
        # 2. BEAM SHAPER
        self.ax.add_patch(Rectangle((2.5, 3), 1, 2, 
                                   facecolor='#4ECDC4', alpha=0.3,
                                   edgecolor='black', linewidth=1))
        self.ax.text(3, 4, 'Beam Shaper', fontsize=20, ha='center', va='center')
        self.ax.text(3, 3, 'Lenses\nBeam\nShaper', fontsize=8, ha='center')
        
        # 3. BBO CRYSTAL (animated oven)
        self.oven = Rectangle((5, 2.5), 2, 2, 
                             facecolor='#FFD166', alpha=0.6,
                             edgecolor='#FF9E00', linewidth=3)
        self.ax.add_patch(self.oven)
        
        # BBO crystal inside
        self.crystal = Rectangle((5.25, 3), 1.5, 1, 
                                facecolor='#06D6A0', alpha=0.8,
                                edgecolor='#048A81', linewidth=2)
        self.ax.add_patch(self.crystal)
        
        # Temperature display
        self.temp_text = self.ax.text(6, 3.8, '25.0°C', fontsize=10, 
                                     ha='center', fontweight='bold',
                                     color='red')
        
        self.ax.text(6, 4.5, ' BBO Crystal', 
                    fontsize=10, ha='center', fontweight='bold')
        
        # 4. QUANTUM EFFECT ZONE
        self.quantum_zone = Rectangle((7.5, 2), 2, 3, 
                                      facecolor='none',
                                      edgecolor='#8338EC', linewidth=2,
                                      linestyle='--')
        self.ax.add_patch(self.quantum_zone)
        self.ax.text(8.5, 5.5, 'QUANTUM\nSQUEEZING\nZONE', 
                    fontsize=9, ha='center', fontweight='bold',
                    color='#8338EC')
        
        # 5. BEAM SPLITTER
        self.bs = Polygon([(10, 3.5), (10.5, 4), (11, 3.5), (10.5, 3)], 
                          facecolor='#8AC926', alpha=0.5,
                          edgecolor='#6A994E', linewidth=2)
        self.ax.add_patch(self.bs)
        self.ax.text(10.5, 4.5, '50/50\nBeam\nSplitter', 
                    fontsize=8, ha='center')
        
        # 6. LOCAL OSCILLATOR
        self.ax.add_patch(Rectangle((10, 6), 1, 0.8, 
                                   facecolor='#118AB2', alpha=0.7,
                                   edgecolor='#073B4C', linewidth=2))
        self.ax.text(10.5, 6.4, 'LO', fontsize=15, ha='center', va='center')
        self.ax.text(10.5, 7, 'Local\nOscillator', fontsize=8, ha='center')
        
        # 7. DETECTORS
        self.detA = Rectangle((12, 2), 1, 1, 
                             facecolor='#FF595E', alpha=0.6,
                             edgecolor='#8B0000', linewidth=2)
        self.detB = Rectangle((12, 4), 1, 1,
                             facecolor='#FF595E', alpha=0.6,
                             edgecolor='#8B0000', linewidth=2)
        self.ax.add_patch(self.detA)
        self.ax.add_patch(self.detB)
        
        self.ax.text(12.5, 2.5, '📊\nDetector\nA', fontsize=8, ha='center')
        self.ax.text(12.5, 4.5, '📊\nDetector\nB', fontsize=8, ha='center')
        
        # 8. SUBTRACTOR
        self.ax.add_patch(Circle((13.5, 3.5), 0.4, 
                                facecolor='#FFD166', alpha=0.7,
                                edgecolor='#FF9E00', linewidth=2))
        self.ax.text(13.5, 3.5, '−', fontsize=20, ha='center', va='center',
                    fontweight='bold')
        self.ax.text(13.5, 2.5, 'Balanced\nSubtractor', 
                    fontsize=8, ha='center')
        
        # 9. OSCILLOSCOPE (move left for visibility)
        self.ax.add_patch(Rectangle((13.2, 4.2), 1.2, 1.7, 
                                   facecolor='#6A4C93', alpha=0.8,
                                   edgecolor='#4A2C73', linewidth=3))
        self.ax.text(13.8, 5.1, 'Oscilloscope', fontsize=20, ha='center', va='center')
        self.ax.text(13.8, 4.3, 'Oscilloscope\n& Analysis', 
                    fontsize=10, ha='center', color='white', fontweight='bold')
        # 10. RANDOM NUMBER DISPLAY (move left)
        self.random_display = Rectangle((13.2, 6.2), 1.2, 0.8, 
                                       facecolor='#06D6A0', alpha=0.7,
                                       edgecolor='#048A81', linewidth=2)
        self.ax.add_patch(self.random_display)
        self.random_text = self.ax.text(13.8, 6.6, 'RANDOM\nBITS', 
                                       fontsize=12, ha='center',
                                       fontweight='bold', color='white')
        
        # Status text
        self.status_text = self.ax.text(7.5, 0.5, 'Initializing...', 
                                       fontsize=12, ha='center',
                                       fontweight='bold', color='darkblue')
        
        # Squeezing meter (move left)
        self.ax.add_patch(Rectangle((11.8, 7.5), 1.3, 0.8, 
                                   facecolor='lightgray', alpha=0.7, zorder=2))
        self.squeeze_text = self.ax.text(12.45, 7.9, 'Squeezing:\n0.0 dB', 
                                        fontsize=11, ha='center', fontweight='bold', zorder=3)
        # Rate display (move left)
        self.ax.add_patch(Rectangle((11.8, 8.3), 1.3, 0.6, 
                                   facecolor='lightgray', alpha=0.7, zorder=2))
        self.rate_text = self.ax.text(12.45, 8.55, 'Rate:\n0 Mbps', 
                                     fontsize=11, ha='center', fontweight='bold', zorder=3)
        
    def update_animation(self, frame):
        """Update animation frame"""
        self.frame = frame
        
        # Clear previous frame elements
        for patch in self.laser_beam:
            patch.remove()
        self.laser_beam = []
        
        for particle in self.squeezed_particles:
            particle.remove()
        self.squeezed_particles = []
        
        for signal in self.detector_signals:
            signal.remove()
        self.detector_signals = []
        
        # Calculate phase for animations
        phase = frame * 2 * np.pi / self.max_frames
        
        # 1. ANIMATE LASER BEAM
        # Green laser beam to BBO
        beam_width = 0.1
        beam_length = 4.5
        
        # Create wavy laser beam
        x_beam = np.linspace(1.5, 5, 50)
        y_beam = 4 + 0.05 * np.sin(10 * x_beam + phase * 10)
        
        # Draw beam with glow effect
        for i in range(10, 0, -1):
            beam = self.ax.plot(x_beam, y_beam, '-', 
                               color='#00FF00', 
                               alpha=0.05 * i,
                               linewidth=10 - i,
                               animated=True)[0]
            self.laser_beam.append(beam)
        
        # Add "photon particles" moving along beam
        if frame % 5 == 0:
            for t in np.linspace(0, 1, 5):
                x_part = 1.5 + t * 3.5
                y_part = 4 + 0.05 * np.sin(10 * x_part + phase * 10)
                particle = Circle((x_part, y_part), 0.05,
                                 color='#00FF00', alpha=0.7,
                                 animated=True)
                self.ax.add_patch(particle)
                self.laser_beam.append(particle)
        
        # 2. ANIMATE BBO CRYSTAL (heat effect)
        # Temperature fluctuation
        temp = 25 + 0.5 * np.sin(phase * 2)
        self.temp_text.set_text(f'{temp:.1f}°C')
        
        # Color change with temperature
        heat_intensity = 0.5 + 0.5 * np.sin(phase)
        self.oven.set_facecolor((1, 0.8, 0.2, 0.3 + 0.3 * heat_intensity))
        
        # 3. ANIMATE SQUEEZING PROCESS
        # Squeezed light coming out of BBO (7-8)
        squeeze_amp = 3.0  # dB squeezing
        x_squeeze = np.linspace(7, 10, 50)
        
        # Normal quantum noise (dotted)
        y_normal = 3.5 + 0.2 * np.sin(10 * x_squeeze + phase * 5)
        normal_beam = self.ax.plot(x_squeeze, y_normal, ':', 
                                  color='#FFA500', alpha=0.5,
                                  linewidth=1, animated=True)[0]
        self.squeezed_particles.append(normal_beam)
        
        # Squeezed quadrature (thin)
        y_squeezed = 3.5 + 0.1 * np.exp(-squeeze_amp/10) * np.sin(10 * x_squeeze + phase * 5)
        squeezed_beam = self.ax.plot(x_squeeze, y_squeezed, '-', 
                                    color='#FF0000', alpha=0.7,
                                    linewidth=1, animated=True)[0]
        self.squeezed_particles.append(squeezed_beam)
        
        # Anti-squeezed quadrature (thick - our randomness source!)
        y_antisqueeze = 3.5 + 0.5 * np.exp(squeeze_amp/10) * np.sin(10 * x_squeeze + phase * 5)
        antisqueeze_beam = self.ax.plot(x_squeeze, y_antisqueeze, '-', 
                                       color='#0000FF', alpha=0.7,
                                       linewidth=3, animated=True)[0]
        self.squeezed_particles.append(antisqueeze_beam)
        
        # Show quantum particles being "squeezed"
        if frame % 3 == 0:
            # Create quantum fluctuation particles
            for i in range(3):
                x_part = 8 + i * 0.5
                y_part = 3.5 + 0.3 * np.random.randn()
                size = 0.03 + 0.07 * np.abs(np.random.randn())
                
                # Color based on quadrature
                if i == 0:
                    color = '#FF0000'  # Squeezed
                    size *= 0.5
                elif i == 1:
                    color = '#FFA500'  # Normal
                else:
                    color = '#0000FF'  # Anti-squeezed
                    size *= 1.5
                
                particle = Circle((x_part, y_part), size,
                                 color=color, alpha=0.6,
                                 animated=True)
                self.ax.add_patch(particle)
                self.squeezed_particles.append(particle)
        
        # 4. ANIMATE BEAM SPLITTER INTERACTION
        # Split beams to detectors
        x_split_A = np.linspace(11, 12, 20)
        y_split_A = 3.5 - 0.01 * (x_split_A - 11) * 10
        
        x_split_B = np.linspace(11, 12, 20)
        y_split_B = 3.5 + 0.01 * (x_split_B - 11) * 10
        
        split_A = self.ax.plot(x_split_A, y_split_A, '-', 
                              color='#00AAFF', alpha=0.7,
                              linewidth=2, animated=True)[0]
        split_B = self.ax.plot(x_split_B, y_split_B, '-', 
                              color='#00AAFF', alpha=0.7,
                              linewidth=2, animated=True)[0]
        self.squeezed_particles.extend([split_A, split_B])
        
        # 5. ANIMATE DETECTOR SIGNALS
        # Create oscillating signals from detectors
        time = np.linspace(12.5, 13.5, 50)
        
        # Detector A signal (sine wave with quantum noise)
        signal_A = 2.5 + 0.3 * np.sin(20 * phase + np.pi/2)
        signal_A += 0.1 * np.random.randn()  # Quantum noise
        
        # Detector B signal (anti-phase)
        signal_B = 4.5 + 0.3 * np.sin(20 * phase - np.pi/2)
        signal_B += 0.1 * np.random.randn()  # Quantum noise
        
        # Draw signal lines
        detector_line_A = self.ax.plot([12.5, 13.1], [signal_A, 3.5], '-',
                                      color='#FF0000', alpha=0.8,
                                      linewidth=1, animated=True)[0]
        detector_line_B = self.ax.plot([12.5, 13.1], [signal_B, 3.5], '-',
                                      color='#FF0000', alpha=0.8,
                                      linewidth=1, animated=True)[0]
        self.detector_signals.extend([detector_line_A, detector_line_B])
        
        # 6. ANIMATE SUBTRACTOR OUTPUT
        # Difference signal
        diff_signal = signal_B - signal_A
        x_output = np.linspace(13.9, 14.5, 30)
        y_output = 3.5 + 0.2 * diff_signal * np.sin(10 * (x_output - 13.9))
        
        output_signal = self.ax.plot(x_output, y_output, '-',
                                    color='#8B00FF', alpha=0.9,
                                    linewidth=2, animated=True)[0]
        self.detector_signals.append(output_signal)
        
        # 7. ANIMATE OSCILLOSCOPE TRACE
        # Create scope trace
        scope_x = np.linspace(14.5, 16, 50)
        scope_y = 3.5 + 0.5 * np.sin(20 * scope_x/1.5 + phase * 10)
        scope_y += 0.2 * diff_signal * np.sin(10 * scope_x/1.5)
        
        scope_trace = self.ax.plot(scope_x, scope_y, '-',
                                  color='#00FF00', alpha=0.9,
                                  linewidth=2, animated=True)[0]
        self.detector_signals.append(scope_trace)
        
        # 8. ANIMATE RANDOM BITS GENERATION
        # Generate random bits based on signal
        if frame % 10 == 0:
            random_bit = '1' if np.random.rand() > 0.5 else '0'
            bit_color = '#00FF00' if random_bit == '1' else '#FF0000'
            
            # Update random bits display
            current_text = self.random_text.get_text()
            if len(current_text.split('\n')[-1]) < 20:
                new_text = current_text + random_bit
            else:
                new_text = 'RANDOM\nBITS\n' + random_bit
            
            self.random_text.set_text(new_text)
            self.random_text.set_color(bit_color)
            
            # Flash the display
            self.random_display.set_facecolor(bit_color + '80')
        
        # 9. UPDATE STATUS
        progress = frame / self.max_frames
        stages = [
            "Initializing Laser...",
            "Pumping BBO Crystal...",
            "Generating Squeezed Light...",
            "Squeezing in Progress...",
            "Measuring Quadratures...",
            "Balanced Detection...",
            "Quantum Noise Analysis...",
            "Generating Random Bits..."
        ]
        
        stage_idx = int(progress * len(stages))
        status = stages[min(stage_idx, len(stages)-1)]
        
        # Add technical details
        squeezing_level = 3.0 + 1.0 * np.sin(phase)
        rate = 100 + 50 * np.abs(np.sin(phase * 2))
        
        self.status_text.set_text(
            f"{status}\n"
            f"Squeezing: {squeezing_level:.1f} dB | "
            f"Rate: {rate:.0f} Mbps | "
            f"Phase: {np.degrees(phase):.0f}°"
        )
        
        # Update meters
        self.squeeze_text.set_text(f'Squeezing:\n{squeezing_level:.1f} dB')
        self.rate_text.set_text(f'Rate:\n{rate:.0f} Mbps')
        
        return self.laser_beam + self.squeezed_particles + self.detector_signals + \
               [self.oven, self.crystal, self.temp_text, self.status_text,
                self.squeeze_text, self.rate_text, self.random_text, self.random_display]
    
    def run_animation(self):
        """Run the complete animation"""
        print(" Starting Quantum Animation...")
        print("Green beam → BBO crystal → Blue squeezed light → Detectors → Random bits!")
        
        # Create animation
        anim = animation.FuncAnimation(
            self.fig, 
            self.update_animation,
            frames=self.max_frames,
            interval=50,  # 20 fps
            blit=True,
            repeat=True
        )
        
        plt.tight_layout()
        plt.show()
        
        # Save animation
        try:
            print(" Saving animation... (This may take a minute)")
            anim.save('quantum_animation.mp4', fps=20, 
                     extra_args=['-vcodec', 'libx264'],
                     dpi=150)
            print(" Animation saved as 'quantum_animation.mp4'")
        except:
            print(" Could not save video. Showing live animation instead.")
        
        return anim

# Run the animation
if __name__ == "__main__":
    animator = QuantumAnimation()
    anim = animator.run_animation()

