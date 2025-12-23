"""
Step-by-step interactive demonstration
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import time

class StepByStepDemo:
    """Interactive step-by-step demonstration"""
    
    def __init__(self):
        self.step = 0
        self.total_steps = 8
        
        # Create figure with subplots for each step
        self.fig = plt.figure(figsize=(18, 10))
        
        # Main animation area
        self.ax_main = plt.axes([0.05, 0.1, 0.65, 0.8])
        self.ax_main.set_xlim(0, 10)
        self.ax_main.set_ylim(0, 10)
        self.ax_main.set_aspect('equal')
        self.ax_main.axis('off')
        
        # Control panel
        self.ax_control = plt.axes([0.73, 0.7, 0.25, 0.25])
        self.ax_control.axis('off')
        
        # Info panel
        self.ax_info = plt.axes([0.73, 0.1, 0.25, 0.55])
        self.ax_info.axis('off')
        
        # Initialize
        self.setup_controls()
        self.show_step(0)
        
    def setup_controls(self):
        """Setup control buttons"""
        
        # Control buttons
        self.ax_prev = plt.axes([0.73, 0.85, 0.1, 0.05])
        self.ax_next = plt.axes([0.85, 0.85, 0.1, 0.05])
        self.ax_auto = plt.axes([0.73, 0.78, 0.22, 0.05])
        
        self.btn_prev = Button(self.ax_prev, '◀ Previous')
        self.btn_next = Button(self.ax_next, 'Next ▶')
        self.btn_auto = Button(self.ax_auto, '🚀 Auto-Run All Steps')
        
        self.btn_prev.on_clicked(self.prev_step)
        self.btn_next.on_clicked(self.next_step)
        self.btn_auto.on_clicked(self.auto_run)
        
        # Parameter sliders
        self.ax_slider1 = plt.axes([0.73, 0.65, 0.22, 0.03])
        self.ax_slider2 = plt.axes([0.73, 0.60, 0.22, 0.03])
        
        self.slider_power = Slider(self.ax_slider1, 'Laser Power', 
                                  50, 500, valinit=200, valstep=10)
        self.slider_temp = Slider(self.ax_slider2, 'BBO Temp (°C)', 
                                20, 30, valinit=25, valstep=0.1)
        
        self.slider_power.on_changed(self.update_params)
        self.slider_temp.on_changed(self.update_params)
        
    def show_step(self, step_num):
        """Show specific step"""
        self.step = step_num
        self.ax_main.clear()
        self.ax_main.set_xlim(0, 10)
        self.ax_main.set_ylim(0, 10)
        self.ax_main.set_aspect('equal')
        self.ax_main.axis('off')
        
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        # Draw step-specific content
        if step_num == 0:
            self.draw_step0()
        elif step_num == 1:
            self.draw_step1()
        elif step_num == 2:
            self.draw_step2()
        elif step_num == 3:
            self.draw_step3()
        elif step_num == 4:
            self.draw_step4()
        elif step_num == 5:
            self.draw_step5()
        elif step_num == 6:
            self.draw_step6()
        elif step_num == 7:
            self.draw_step7()
        
        self.fig.canvas.draw_idle()
    
    def draw_step0(self):
        """Step 0: Introduction"""
        self.ax_main.text(5, 8, 'BBO SQUEEZED LIGHT QRNG', 
                         fontsize=24, ha='center', fontweight='bold',
                         color='darkblue')
        
        self.ax_main.text(5, 6, 'Complete Process Demonstration', 
                         fontsize=18, ha='center', color='darkred')
        
        # Draw simplified setup
        components = [
            (1, 5, '🔦', 'Laser'),
            (3, 5, '📐', 'Lenses'),
            (5, 5, '🔥💎', 'BBO'),
            (7, 5, '📊📊', 'Detectors'),
            (9, 5, '🎲', 'Random\nBits'),
        ]
        
        for x, y, emoji, label in components:
            self.ax_main.add_patch(plt.Circle((x, y), 0.5, 
                                            facecolor='lightblue',
                                            edgecolor='black', alpha=0.5))
            self.ax_main.text(x, y, emoji, fontsize=20, 
                            ha='center', va='center')
            self.ax_main.text(x, y-1, label, fontsize=10, 
                            ha='center', fontweight='bold')
            
        # Connect with arrows
        for i in range(len(components)-1):
            x1, y1 = components[i][0], components[i][1]
            x2, y2 = components[i+1][0], components[i+1][1]
            self.ax_main.arrow(x1+0.5, y1, x2-x1-1, 0, 
                             head_width=0.2, head_length=0.2,
                             fc='green', ec='green', alpha=0.7)
        
        # Info text
        info_text = """
        Welcome to the BBO Squeezed Light QRNG Demo!
        
        This interactive demonstration will show you:
        
        1. Laser beam generation and shaping
        2. BBO crystal nonlinear effects
        3. Quantum squeezing creation
        4. Homodyne detection
        5. Random bit extraction
        
        Click 'Next ▶' to begin!
        """
        self.ax_info.text(0, 0.9, info_text, fontsize=12,
                         va='top', ha='left',
                         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    def draw_step1(self):
        """Step 1: Laser Generation"""
        self.ax_main.text(5, 9, 'STEP 1: LASER GENERATION', 
                         fontsize=20, ha='center', fontweight='bold',
                         color='darkgreen')
        
        # Draw laser
        laser_box = plt.Rectangle((3, 4), 2, 2, 
                                 facecolor='#FF6B6B',
                                 edgecolor='black', linewidth=3)
        self.ax_main.add_patch(laser_box)
        self.ax_main.text(4, 5, '🔦', fontsize=40, ha='center', va='center')
        self.ax_main.text(4, 3.5, '532 nm\nGreen Laser\n200 mW', 
                         fontsize=12, ha='center', fontweight='bold')
        
        # Animated beam
        x_beam = np.linspace(5, 7, 100)
        y_beam = 5 + 0.1 * np.sin(10 * x_beam)
        
        for i in range(5, 0, -1):
            self.ax_main.plot(x_beam, y_beam, '-', 
                            color='#00FF00', 
                            alpha=0.1 * i,
                            linewidth=8 - i,
                            animated=True)
        
        # Photon particles
        for i in range(10):
            x_part = 5 + i * 0.2
            y_part = 5 + 0.1 * np.sin(10 * x_part)
            self.ax_main.add_patch(plt.Circle((x_part, y_part), 0.05,
                                            color='#00FF00', alpha=0.7))
        
        # Info text
        info_text = """
        LASER GENERATION:
        
        • Wavelength: 532 nm (green)
        • Power: 200 mW (adjustable)
        • Type: Continuous wave laser
        
        The laser provides the pump energy needed
        for nonlinear effects in the BBO crystal.
        
        Photons travel as coherent waves with
        well-defined phase and amplitude.
        """
        self.ax_info.text(0, 0.9, info_text, fontsize=12,
                         va='top', ha='left',
                         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    def draw_step2(self):
        """Step 2: Beam Shaping"""
        self.ax_main.text(5, 9, 'STEP 2: BEAM SHAPING', 
                         fontsize=20, ha='center', fontweight='bold',
                         color='darkgreen')
        
        # Draw lenses
        x_positions = [2, 3, 4]
        lens_colors = ['#4ECDC4', '#45B7AA', '#3DA194']
        
        for i, x in enumerate(x_positions):
            # Lens shape (exaggerated)
            lens_x = [x, x+0.5, x+0.5, x]
            lens_y = [4, 4.5, 5.5, 6]
            self.ax_main.fill(lens_x, lens_y, lens_colors[i], alpha=0.6)
            self.ax_main.plot(lens_x, lens_y, 'black', linewidth=1)
            
            if i == 1:
                self.ax_main.text(x+0.25, 5, '🔍', fontsize=30,
                                ha='center', va='center')
        
        # Beam before/after
        # Before (diverging)
        x_before = np.linspace(1, 2, 50)
        y_before = 5 + 0.3 * (x_before - 1.5)
        self.ax_main.plot(x_before, y_before, ':', color='red', alpha=0.5)
        self.ax_main.plot(x_before, 10-y_before, ':', color='red', alpha=0.5)
        
        # After (collimated)
        x_after = np.linspace(4.5, 6, 50)
        self.ax_main.plot(x_after, [5.1]*50, '-', color='green', linewidth=2)
        self.ax_main.plot(x_after, [4.9]*50, '-', color='green', linewidth=2)
        
        # Labels
        self.ax_main.text(1.5, 6.5, 'Diverging\nBeam', fontsize=10,
                         ha='center', color='red')
        self.ax_main.text(5, 5.5, 'Collimated\nBeam', fontsize=10,
                         ha='center', color='green')
        
        # Info text
        info_text = """
        BEAM SHAPING:
        
        Purpose: Focus laser to small spot in BBO crystal
        
        • Lenses collimate and focus beam
        • Smaller spot = higher intensity
        • Better nonlinear conversion efficiency
        
        Typical beam parameters:
        • Waist: 50-100 μm in crystal
        • Rayleigh length: optimized for crystal length
        • Mode: TEM₀₀ Gaussian
        """
        self.ax_info.text(0, 0.9, info_text, fontsize=12,
                         va='top', ha='left',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    def draw_step3(self):
        """Step 3: BBO Nonlinear Interaction"""
        self.ax_main.text(5, 9, 'STEP 3: BBO NONLINEAR INTERACTION', 
                         fontsize=20, ha='center', fontweight='bold',
                         color='darkgreen')
        
        # Draw BBO crystal
        crystal = plt.Rectangle((4, 3), 2, 4, 
                               facecolor='#06D6A0', alpha=0.7,
                               edgecolor='#048A81', linewidth=3)
        self.ax_main.add_patch(crystal)
        
        # Temperature indicator
        self.ax_main.add_patch(plt.Rectangle((3.8, 2.8), 2.4, 0.4,
                                           facecolor='#FF6B6B',
                                           edgecolor='#FF0000', linewidth=2))
        self.ax_main.text(5, 3, f'{self.slider_temp.val:.1f}°C', 
                         fontsize=14, ha='center', va='center',
                         fontweight='bold', color='white')
        
        self.ax_main.text(5, 5, '💎 BBO\nCrystal', fontsize=20,
                         ha='center', va='center')
        
        # Show pump beam entering
        x_in = np.linspace(2, 4, 50)
        y_in = 5 + 0.1 * np.sin(10 * x_in)
        self.ax_main.plot(x_in, y_in, '-', color='#00FF00', linewidth=2)
        
        # Show generated beams (multiple wavelengths)
        x_out = np.linspace(6, 8, 50)
        
        # Pump (green, depleted)
        y_pump = 5 + 0.05 * np.sin(10 * x_out)
        self.ax_main.plot(x_out, y_pump, '-', color='#00FF00', 
                         linewidth=1, alpha=0.5)
        
        # Signal (red, 1064 nm)
        y_signal = 5.5 + 0.2 * np.sin(10 * x_out + np.pi/4)
        self.ax_main.plot(x_out, y_signal, '-', color='#FF0000', 
                         linewidth=2)
        
        # Idler (also red, 1064 nm)
        y_idler = 4.5 + 0.2 * np.sin(10 * x_out - np.pi/4)
        self.ax_main.plot(x_out, y_idler, '-', color='#FF0000', 
                         linewidth=2)
        
        # Labels
        self.ax_main.text(3, 5.5, '532 nm\nPump', fontsize=10,
                         ha='right', color='green')
        self.ax_main.text(7, 5.8, '1064 nm\nSignal', fontsize=10,
                         ha='left', color='red')
        self.ax_main.text(7, 4.2, '1064 nm\nIdler', fontsize=10,
                         ha='left', color='red')
        
        # Info text
        info_text = f"""
        BBO NONLINEAR INTERACTION:
        
        Process: Optical Parametric Down-Conversion
        
        • 532 nm photon → 1064 nm photon pair
        • Energy conserved: E_pump = E_signal + E_idler
        • Momentum conserved: phase matching
        
        BBO Parameters:
        • Temperature: {self.slider_temp.val:.1f}°C
        • Length: 5 mm
        • Type I phase matching: o + o → e
        
        Critical: Temperature controls phase matching!
        """
        self.ax_info.text(0, 0.9, info_text, fontsize=12,
                         va='top', ha='left',
                         bbox=dict(boxstyle='round', facecolor='#06D6A0', alpha=0.3))
    
    def draw_step4(self):
        """Step 4: Quantum Squeezing Creation"""
        self.ax_main.text(5, 9, 'STEP 4: QUANTUM SQUEEZING CREATION', 
                         fontsize=20, ha='center', fontweight='bold',
                         color='darkgreen')
        
        # Draw phase space representation
        # Create circle representing uncertainty
        angles = np.linspace(0, 2*np.pi, 100)
        circle_x = 5 + np.cos(angles)
        circle_y = 5 + np.sin(angles)
        self.ax_main.plot(circle_x, circle_y, '--', color='gray', alpha=0.5)
        self.ax_main.text(6.5, 5.8, 'Shot Noise\nLimit', fontsize=10,
                         color='gray')
        
        # Draw squeezed ellipse
        squeeze_factor = 0.5  # 3 dB squeezing
        ellipse_x = 5 + squeeze_factor * np.cos(angles)
        ellipse_y = 5 + (1/squeeze_factor) * np.sin(angles)
        
        # Rotate ellipse to show different quadratures
        theta = np.pi/4  # 45 degrees rotation
        rot_x = ellipse_x * np.cos(theta) - ellipse_y * np.sin(theta)
        rot_y = ellipse_x * np.sin(theta) + ellipse_y * np.cos(theta)
        
        self.ax_main.fill(rot_x, rot_y, color='blue', alpha=0.3)
        self.ax_main.plot(rot_x, rot_y, '-', color='darkblue', linewidth=2)
        
        # Draw axes
        self.ax_main.arrow(5, 5, 2, 0, head_width=0.1, head_length=0.2,
                          fc='black', ec='black')
        self.ax_main.arrow(5, 5, 0, 2, head_width=0.1, head_length=0.2,
                          fc='black', ec='black')
        
        self.ax_main.text(7, 4.8, 'X-quadrature\n(Amplitude)', fontsize=10)
        self.ax_main.text(4.8, 7, 'P-quadrature\n(Phase)', fontsize=10)
        
        # Show squeezing directions
        self.ax_main.text(4, 4, 'Squeezed\n(less noise)', fontsize=10,
                         ha='right', color='red')
        self.ax_main.text(6, 6, 'Anti-squeezed\n(more noise → randomness!)', 
                         fontsize=10, ha='left', color='blue')
        
        # Info text
        info_text = """
        QUANTUM SQUEEZING:
        
        What is squeezing?
        • Reduce quantum noise in one quadrature
        • Increase noise in conjugate quadrature
        • Total uncertainty still obeys Heisenberg
        
        Visual representation:
        • Circle → Shot noise limit (normal light)
        • Ellipse → Squeezed light
        • Thin direction → Less noise (squeezed)
        • Thick direction → More noise (anti-squeezed)
        
        For QRNG: We use the ANTI-SQUEEZED
        quadrature as our randomness source!
        """
        self.ax_info.text(0, 0.9, info_text, fontsize=12,
                         va='top', ha='left',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    def draw_step5(self):
        """Step 5: Homodyne Detection"""
        self.ax_main.text(5, 9, 'STEP 5: HOMODYNE DETECTION', 
                         fontsize=20, ha='center', fontweight='bold',
                         color='darkgreen')
        
        # Draw beam splitter
        bs_x = [4, 4.5, 5, 4.5]
        bs_y = [4, 4.5, 4, 3.5]
        self.ax_main.fill(bs_x, bs_y, color='#8AC926', alpha=0.5)
        self.ax_main.text(4.5, 4, '↔', fontsize=30, ha='center', va='center')
        self.ax_main.text(4.5, 3, '50/50\nBeam Splitter', fontsize=10,
                         ha='center')
        
        # Draw Local Oscillator
        self.ax_main.add_patch(plt.Rectangle((3, 6), 1, 0.5,
                                           facecolor='#118AB2',
                                           edgecolor='#073B4C', linewidth=2))
        self.ax_main.text(3.5, 6.25, '💡', fontsize=20,
                         ha='center', va='center')
        self.ax_main.text(3.5, 7, 'Local\nOscillator', fontsize=10,
                         ha='center')
        
        # Draw beams
        # LO to BS
        self.ax_main.plot([3.5, 4.25], [6.5, 4.25], '--', 
                         color='orange', linewidth=2)
        
        # Squeezed light to BS
        self.ax_main.plot([2, 4.25], [4, 4.25], '--', 
                         color='blue', linewidth=2)
        
        # Output beams
        self.ax_main.plot([4.75, 6], [4.75, 5.5], '-', 
                         color='purple', linewidth=2)
        self.ax_main.plot([4.75, 6], [3.75, 2.5], '-', 
                         color='purple', linewidth=2)
        
        # Detectors
        self.ax_main.add_patch(plt.Circle((6.5, 5.5), 0.4,
                                         facecolor='#FF595E',
                                         edgecolor='#8B0000', linewidth=2))
        self.ax_main.add_patch(plt.Circle((6.5, 2.5), 0.4,
                                         facecolor='#FF595E',
                                         edgecolor='#8B0000', linewidth=2))
        
        self.ax_main.text(6.5, 5.5, 'A', fontsize=16,
                         ha='center', va='center', fontweight='bold')
        self.ax_main.text(6.5, 2.5, 'B', fontsize=16,
                         ha='center', va='center', fontweight='bold')
        
        # Subtraction
        self.ax_main.add_patch(plt.Circle((7.5, 4), 0.3,
                                         facecolor='#FFD166',
                                         edgecolor='#FF9E00', linewidth=2))
        self.ax_main.text(7.5, 4, '−', fontsize=20,
                         ha='center', va='center', fontweight='bold')
        
        # Output signal
        x_signal = np.linspace(8, 9, 50)
        y_signal = 4 + 0.5 * np.sin(10 * x_signal)
        self.ax_main.plot(x_signal, y_signal, '-', color='green', linewidth=2)
        
        # Labels
        self.ax_main.text(2.5, 4.5, 'Squeezed\nLight', fontsize=10,
                         ha='right', color='blue')
        self.ax_main.text(3, 5.5, 'Reference\nLaser', fontsize=10,
                         ha='right', color='orange')
        self.ax_main.text(8.5, 4.5, 'Quantum\nNoise\nSignal', fontsize=10,
                         ha='left', color='green')
        
        # Info text
        info_text = """
        HOMODYNE DETECTION:
        
        Purpose: Measure quantum noise precisely
        
        How it works:
        1. Squeezed light + Reference laser → Beam splitter
        2. Two output beams hit detectors A and B
        3. Subtract signals: A - B
        
        Why subtract?
        • Cancels classical laser noise
        • Amplifies quantum noise difference
        • Common-mode rejection
        
        Result: Pure quantum noise signal
        (our randomness source!)
        """
        self.ax_info.text(0, 0.9, info_text, fontsize=12,
                         va='top', ha='left',
                         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    def draw_step6(self):
        """Step 6: Signal Processing"""
        self.ax_main.text(5, 9, 'STEP 6: SIGNAL PROCESSING', 
                         fontsize=20, ha='center', fontweight='bold',
                         color='darkgreen')
        
        # Draw oscilloscope
        scope_box = plt.Rectangle((3, 2), 4, 4,
                                 facecolor='#6A4C93',
                                 edgecolor='#4A2C73', linewidth=3)
        self.ax_main.add_patch(scope_box)
        
        # Grid
        for i in range(1, 4):
            self.ax_main.plot([3, 7], [2+i, 2+i], ':', 
                             color='white', alpha=0.3)
            self.ax_main.plot([3+i, 3+i], [2, 6], ':', 
                             color='white', alpha=0.3)
        
        # Draw signal trace
        x_trace = np.linspace(3, 7, 200)
        y_trace = 4 + 1.5 * np.sin(10 * x_trace + self.step * 0.1)
        y_trace += 0.5 * np.random.randn(200)  # Quantum noise
        
        self.ax_main.plot(x_trace, y_trace, '-', color='#00FF00', 
                         linewidth=2, alpha=0.9)
        
        # Histogram on the side
        hist_x = np.linspace(7.5, 9, 100)
        hist_y = 2 + 4 * np.exp(-0.5 * ((hist_x - 8.25)/0.5)**2)
        self.ax_main.fill_between(hist_x, 2, hist_y, 
                                 color='blue', alpha=0.3)
        self.ax_main.plot(hist_x, hist_y, '-', color='darkblue', linewidth=2)
        
        # Gaussian fit
        self.ax_main.text(8.25, 5, 'Gaussian\nDistribution', fontsize=10,
                         ha='center', color='darkblue')
        
        # Labels
        self.ax_main.text(5, 1.5, 'Oscilloscope\n(Noise vs Time)', 
                         fontsize=12, ha='center', color='white',
                         bbox=dict(boxstyle='round', facecolor='#6A4C93'))
        self.ax_main.text(8.25, 1.5, 'Probability\nDistribution', 
                         fontsize=12, ha='center', color='darkblue',
                         bbox=dict(boxstyle='round', facecolor='lightblue'))
        
        # ADC quantization
        adc_x = [1, 1.5, 1.5, 1]
        adc_y = [2, 2.5, 5.5, 6]
        self.ax_main.fill(adc_x, adc_y, color='#FF6B6B', alpha=0.5)
        self.ax_main.text(1.25, 4, 'ADC\nQuantization', fontsize=10,
                         ha='center', va='center')
        
        # Show quantization levels
        for i in range(8):
            y_level = 2 + i * 0.5
            self.ax_main.plot([1.5, 1.8], [y_level, y_level], 
                             color='black', linewidth=1)
            self.ax_main.text(1.9, y_level, f'{i:03b}', fontsize=8,
                             va='center')
        
        # Info text
        info_text = """
        SIGNAL PROCESSING:
        
        1. Analog-to-Digital Conversion:
           • Continuous signal → Discrete samples
           • 8-bit ADC = 256 levels
           • Sampling rate: 200 MSamples/s
        
        2. Signal Analysis:
           • Time domain: Quantum noise fluctuations
           • Frequency domain: Bandwidth characterization
           • Statistics: Gaussian distribution check
        
        3. Security Analysis:
           • Calculate min-entropy bounds
           • Monitor for attacks or anomalies
           • Ensure quantum origin of randomness
        """
        self.ax_info.text(0, 0.9, info_text, fontsize=12,
                         va='top', ha='left',
                         bbox=dict(boxstyle='round', facecolor='#6A4C93', alpha=0.2))
    
    def draw_step7(self):
        """Step 7: Random Bit Generation"""
        self.ax_main.text(5, 9, 'STEP 7: RANDOM BIT GENERATION', 
                         fontsize=20, ha='center', fontweight='bold',
                         color='darkgreen')
        
        # Draw random bit stream
        bits = np.random.randint(0, 2, 64)
        
        # Display as 8x8 grid
        for i in range(8):
            for j in range(8):
                idx = i * 8 + j
                bit = bits[idx]
                x = 3 + j * 0.5
                y = 6 - i * 0.5
                
                color = '#00FF00' if bit == 1 else '#FF0000'
                self.ax_main.add_patch(plt.Rectangle((x, y), 0.4, 0.4,
                                                   facecolor=color,
                                                   edgecolor='black',
                                                   linewidth=1))
                self.ax_main.text(x+0.2, y+0.2, str(bit), fontsize=10,
                                ha='center', va='center',
                                fontweight='bold', color='white')
        
        # Rate display
        self.ax_main.add_patch(plt.Rectangle((3, 2), 4, 1,
                                           facecolor='lightgray',
                                           edgecolor='black', linewidth=2))
        self.ax_main.text(5, 2.5, f'Bit Rate: {250+50*np.random.rand():.0f} Mbps',
                         fontsize=16, ha='center', fontweight='bold')
        
        # NIST test results
        tests = [
            ('Frequency', 'PASS ✓', 0.52),
            ('Block Frequency', 'PASS ✓', 0.48),
            ('Runs', 'PASS ✓', 0.61),
            ('Longest Run', 'PASS ✓', 0.43),
            ('FFT', 'PASS ✓', 0.55),
            ('Serial', 'PASS ✓', 0.49),
        ]
        
        y_pos = 3.5
        for test_name, result, p_value in tests:
            color = 'green' if 'PASS' in result else 'red'
            self.ax_main.text(1, y_pos, test_name, fontsize=10,
                             ha='left')
            self.ax_main.text(3.5, y_pos, result, fontsize=10,
                             ha='left', color=color, fontweight='bold')
            self.ax_main.text(5, y_pos, f'p={p_value:.2f}', fontsize=10,
                             ha='left')
            y_pos -= 0.4
        
        # Final output
        self.ax_main.add_patch(plt.Rectangle((7, 3), 2, 2,
                                           facecolor='#06D6A0',
                                           edgecolor='#048A81', linewidth=3))
        self.ax_main.text(8, 4, 'SECURE\nRANDOM\nBITS', fontsize=14,
                         ha='center', va='center',
                         fontweight='bold', color='white')
        
        # Info text
        info_text = """
        RANDOM BIT GENERATION:
        
        Final Steps:
        
        1. Entropy Extraction:
           • Apply Toeplitz hashing
           • Extract secure bits from noise
           • Remove any residual bias
        
        2. Statistical Testing:
           • NIST test suite
           • Diehard tests
           • TestU01 battery
        
        3. Security Certification:
           • Calculate min-entropy
           • Verify against SDI bounds
           • Ensure device-independence
        
        Performance:
        • Rate: 200-500 Mbps expected
        • Security: Semi-device-independent
        • Quality: True quantum randomness
        """
        self.ax_info.text(0, 0.9, info_text, fontsize=12,
                         va='top', ha='left',
                         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    def prev_step(self, event):
        """Go to previous step"""
        self.step = (self.step - 1) % self.total_steps
        self.show_step(self.step)
    
    def next_step(self, event):
        """Go to next step"""
        self.step = (self.step + 1) % self.total_steps
        self.show_step(self.step)
    
    def auto_run(self, event):
        """Auto-run through all steps"""
        for i in range(self.total_steps):
            self.show_step(i)
            plt.pause(2)  # Pause 2 seconds between steps
    
    def update_params(self, val):
        """Update parameters from sliders"""
        self.show_step(self.step)  # Redraw with new parameters
    
    def run(self):
        """Run the demo"""
        print("🎮 Starting Interactive Demonstration...")
        print("Use buttons to navigate through each step!")
        plt.show()

# Run the demo
if __name__ == "__main__":
    demo = StepByStepDemo()
    demo.run()