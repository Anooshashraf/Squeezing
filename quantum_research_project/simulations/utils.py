"""
Utility functions
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arrow
import json

def save_results(results, filename='simulation_results.json'):
    """Save simulation results to JSON"""
    with open(filename, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            else:
                return obj
        
        serializable_results = {k: convert(v) for k, v in results.items()}
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {filename}")

def load_results(filename='simulation_results.json'):
    """Load simulation results from JSON"""
    with open(filename, 'r') as f:
        results = json.load(f)
    
    print(f"Results loaded from {filename}")
    return results

def create_optical_diagram():
    """Create a visual diagram of the optical setup"""
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Set limits
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    ax.text(5, 5.5, 'BBO Squeezed Light QRNG Setup', 
           fontsize=16, fontweight='bold', ha='center')
    
    # Components
    components = {
        'Laser': (1, 3, '📟', '532 nm Laser'),
        'Beam Shaper': (2.5, 3, '🔍', 'Lenses'),
        'BBO Oven': (4, 3, '🔥💎', 'BBO Crystal\nin Oven'),
        'Beam Splitter': (6, 3, '↔', '50/50\nBeam Splitter'),
        'LO Path': (6, 4.5, '💡', 'Local\nOscillator'),
        'Detectors': (7.5, 2.5, '📊', 'Detector A'),
        ' ': (7.5, 3.5, '📊', 'Detector B'),
        'Subtractor': (8.5, 3, '➖', 'Balanced\nDetection'),
        'OSC': (9.5, 3, '📈', 'Oscilloscope'),
    }
    
    # Draw components
    for name, (x, y, emoji, label) in components.items():
        # Box
        rect = Rectangle((x-0.4, y-0.4), 0.8, 0.8, 
                        fill=True, color='lightblue', alpha=0.3)
        ax.add_patch(rect)
        
        # Emoji
        ax.text(x, y+0.1, emoji, fontsize=20, ha='center', va='center')
        
        # Label
        ax.text(x, y-0.6, label, fontsize=8, ha='center', va='center')
    
    # Arrows (light path)
    arrows = [
        (1.4, 3, 2.1, 3),  # Laser to lenses
        (2.9, 3, 3.6, 3),  # Lenses to BBO
        (4.4, 3, 5.6, 3),  # BBO to beam splitter
        (6, 3.4, 6, 4.1),  # BS to LO path
        (6.4, 3, 7.1, 2.5), # BS to detector A
        (6.4, 3, 7.1, 3.5), # BS to detector B
        (7.9, 2.5, 8.1, 2.8), # Det A to subtractor
        (7.9, 3.5, 8.1, 3.2), # Det B to subtractor
        (8.9, 3, 9.1, 3),  # Subtractor to OSC
    ]
    
    for x1, y1, x2, y2 in arrows:
        ax.arrow(x1, y1, x2-x1, y2-y1, 
                head_width=0.1, head_length=0.1, 
                fc='red', ec='red', alpha=0.7)
    
    # Labels for paths
    ax.text(2, 3.3, 'Pump Beam\n(532 nm)', fontsize=8, ha='center')
    ax.text(5, 3.3, 'Squeezed Light\n(1064 nm)', fontsize=8, ha='center')
    ax.text(6.5, 4.7, 'Reference Laser', fontsize=8, ha='center')
    
    # Phase control
    phase_circle = Circle((6, 4.5), 0.2, fill=True, color='yellow', alpha=0.5)
    ax.add_patch(phase_circle)
    ax.text(6, 4.5, 'φ', fontsize=10, ha='center', va='center', fontweight='bold')
    ax.text(6, 5, 'Phase\nControl', fontsize=8, ha='center')
    
    plt.tight_layout()
    plt.savefig('optical_setup_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def print_summary(results_dict):
    """Print a nice summary of simulation results"""
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    
    for scenario, results in results_dict.items():
        print(f"\n📊 {scenario}:")
        print(f"  Squeezing: {results['parameters']['squeezing_db']:.1f} dB")
        print(f"  Secure bits/sample: {results['secure_bits_per_sample']:.3f}")
        print(f"  Final rate: {results['rate_mbps']:.1f} Mbps")
        
        # Compare with paper
        paper_rate = 580.7
        percentage = (results['rate_mbps'] / paper_rate) * 100
        print(f"  vs Paper ({paper_rate} Mbps): {percentage:.1f}%")
        
        if percentage > 70:
            print(f"  ✅ Excellent! Close to paper's performance")
        elif percentage > 50:
            print(f"  ⚠️  Good, but room for improvement")
        else:
            print(f"  ❌ Needs optimization")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)
    
    # Find best scenario
    best_scenario = max(results_dict.items(), 
                       key=lambda x: x[1]['rate_mbps'])
    
    print(f"1. Target {best_scenario[1]['parameters']['squeezing_db']:.1f} dB squeezing")
    print(f"2. Keep LO noise < {best_scenario[1]['parameters']['LO_noise_fraction']:.3f}")
    print(f"3. Electronic noise < {best_scenario[1]['parameters']['electronic_noise_db']} dB")
    print(f"4. Expected rate: {best_scenario[1]['rate_mbps']:.1f} Mbps")
    
    if best_scenario[1]['rate_mbps'] > 300:
        print("\n🎯 RESULT: BBO is viable for SDI-QRNG!")
    else:
        print("\n⚠️  WARNING: May need PPKTP for higher rates")