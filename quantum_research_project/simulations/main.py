"""
Main simulation script - Run everything!
"""
import numpy as np
import matplotlib.pyplot as plt
from bbo_simulation import BBOSqueezer
from detection_simulation import HomodyneDetector
from security_simulation import SDIQRNGSimulator
from utils import save_results, create_optical_diagram, print_summary

def run_complete_simulation():
    """
    Run all simulations
    """
    print("🚀 Starting BBO Squeezed Light QRNG Simulation")
    print("="*60)
    
    # Initialize simulators
    bbo_sim = BBOSqueezer()
    detector_sim = HomodyneDetector()
    security_sim = SDIQRNGSimulator()
    
    # Store all results
    all_results = {}
    
    # 1. BBO Optical Simulation
    print("\n1️⃣  Running BBO Optical Simulations...")
    bbo_results = {}
    
    # Phase matching
    temps, efficiency, delta_k = bbo_sim.phase_matching_curve()
    bbo_results['phase_matching'] = {
        'temperatures': temps,
        'efficiency': efficiency,
        'delta_k': delta_k
    }
    
    # Single-pass vs OPO comparison
    print("   Comparing single-pass vs OPO...")
    pump_power = 200e-3  # 200 mW
    crystal_length = 5e-3  # 5 mm
    
    single_pass = bbo_sim.calculate_squeezing(
        pump_power, crystal_length, method='single_pass')
    
    cavity_params = {'finesse': 100, 'loss': 0.01, 'output_coupling': 0.1}
    opo = bbo_sim.calculate_squeezing(
        pump_power, crystal_length, method='OPO', cavity_params=cavity_params)
    
    bbo_results['comparison'] = {
        'single_pass': single_pass,
        'OPO': opo
    }
    
    # Walk-off analysis
    walkoff = bbo_sim.simulate_walkoff(crystal_length)
    bbo_results['walkoff'] = walkoff
    
    all_results['bbo'] = bbo_results
    
    # Plot BBO results
    print("   Generating BBO plots...")
    bbo_sim.plot_simulation_results()
    
    # 2. Detection Simulation
    print("\n2️⃣  Running Detection Simulations...")
    detector_results = {}
    
    # Phase scan
    phases, variances = detector_sim.simulate_detection(
        squeezing_db=3.5, LO_power=10, phase_scan=True)
    detector_results['phase_scan'] = {
        'phases': phases,
        'variances': variances
    }
    
    # SNR analysis
    LO_powers = np.logspace(-1, 2, 20)  # 0.1 to 100 mW
    snr_data = []
    
    for P in LO_powers:
        SNR, details = detector_sim.calculate_SNR(3.5, P)
        snr_data.append({
            'LO_power': P,
            'SNR': SNR,
            'details': details
        })
    
    detector_results['snr_analysis'] = snr_data
    
    all_results['detection'] = detector_results
    
    # Plot detection results
    print("   Generating detection plots...")
    detector_sim.plot_detection_results(squeezing_db=3.5, LO_power=10)
    
    # 3. Security Simulation
    print("\n3️⃣  Running Security Simulations...")
    security_results = security_sim.simulate_attack_scenarios(base_squeezing=3.5)
    all_results['security'] = security_results
    
    # Plot security results
    print("   Generating security plots...")
    security_sim.plot_performance_analysis()
    
    # 4. Generate diagrams
    print("\n4️⃣  Generating optical diagrams...")
    create_optical_diagram()
    
    # 5. Save results
    print("\n💾 Saving all results...")
    save_results(all_results)
    
    # 6. Print summary
    print("\n" + "="*60)
    print("📊 FINAL SIMULATION SUMMARY")
    print("="*60)
    
    print_summary(security_results)
    
    # 7. Key findings
    print("\n" + "="*60)
    print("🔑 KEY FINDINGS")
    print("="*60)
    
    # Extract key numbers
    single_pass_squeezing = single_pass['squeezing_db']
    opo_squeezing = opo['squeezing_db']
    
    print(f"1. BBO Performance:")
    print(f"   - Single-pass: {single_pass_squeezing:.1f} dB squeezing")
    print(f"   - OPO: {opo_squeezing:.1f} dB squeezing")
    print(f"   - Recommended: {'OPO' if opo_squeezing > 3 else 'Single-pass'}")
    
    # Best security scenario
    best_rate = max([r['rate_mbps'] for r in security_results.values()])
    best_scenario = [k for k, v in security_results.items() 
                    if v['rate_mbps'] == best_rate][0]
    
    print(f"\n2. QRNG Performance:")
    print(f"   - Best scenario: {best_scenario}")
    print(f"   - Maximum rate: {best_rate:.1f} Mbps")
    
    # Comparison with paper
    paper_rate = 580.7
    if best_rate > paper_rate * 0.7:
        print(f"   - ✅ Close to paper's {paper_rate} Mbps!")
    else:
        print(f"   - ⚠️  {best_rate/paper_rate*100:.1f}% of paper's rate")
    
    print(f"\n3. Experimental Requirements:")
    print(f"   - Temperature stability: ±0.1°C")
    print(f"   - LO power: ~10 mW")
    print(f"   - Phase stability: < 1°")
    print(f"   - Electronic noise: < -13 dB")
    
    print("\n🎯 SIMULATION COMPLETE! Ready for experimental work!")
    
    return all_results

if __name__ == "__main__":
    # Run everything
    results = run_complete_simulation()
    
    # Quick export option
    export = input("\nExport quick report? (y/n): ")
    if export.lower() == 'y':
        with open('simulation_report.txt', 'w') as f:
            f.write("BBO Squeezed Light QRNG Simulation Report\n")
            f.write("="*50 + "\n\n")
            
            # Write key findings
            f.write("KEY FINDINGS:\n")
            f.write("-"*30 + "\n")
            
            # Get values from results
            bbo_data = results['bbo']['comparison']
            single_pass_db = bbo_data['single_pass']['squeezing_db']
            opo_db = bbo_data['OPO']['squeezing_db']
            
            f.write(f"Single-pass BBO squeezing: {single_pass_db:.2f} dB\n")
            f.write(f"OPO BBO squeezing: {opo_db:.2f} dB\n")
            
            # Best security result
            security_data = results['security']
            best_scenario = max(security_data.items(), 
                              key=lambda x: x[1]['rate_mbps'])
            
            f.write(f"\nBest QRNG performance:\n")
            f.write(f"  Scenario: {best_scenario[0]}\n")
            f.write(f"  Rate: {best_scenario[1]['rate_mbps']:.1f} Mbps\n")
            f.write(f"  Secure bits/sample: {best_scenario[1]['secure_bits_per_sample']:.3f}\n")
            
            f.write("\nCONCLUSION:\n")
            f.write("-"*30 + "\n")
            if best_scenario[1]['rate_mbps'] > 300:
                f.write("✅ BBO is viable for SDI-QRNG! Proceed with experiment.\n")
            else:
                f.write("⚠️  Consider PPKTP for better performance.\n")
        
        print("Report saved to 'simulation_report.txt'")