from security_simulation import SDIQRNGSimulator

sim = SDIQRNGSimulator()

print("="*70)
print("VERIFICATION TEST - Matching Paper's Results")
print("="*70)

# Test 1: Try to match paper's conditions
print("\n1. Paper's conditions (PPKTP):")
print("   Squeezing: 3.8-6.5 dB (use 6 dB)")
print("   LO noise: ~0.01 (low)")
print("   Electronic noise: -15 dB (good detector)")
print("   Expected: H_min ≈ 7.21 bits, rate ≈ 580.7 Mbps")

res_paper = sim.simulate_complete_protocol(6.0, 0.01, -15)
print(f"\n   Simulated:")
print(f"   • H_min_smooth: {res_paper['H_min_smooth']:.2f} bits")
print(f"   • Secure bits/sample: {res_paper['secure_bits_per_sample']:.3f}")
print(f"   • Rate: {res_paper['rate_mbps']:.1f} Mbps")
print(f"   • Paper's rate: 580.7 Mbps")
print(f"   • Match: {res_paper['rate_mbps']/580.7*100:.1f}%")

# Test 2: BBO realistic scenario
print("\n2. BBO realistic scenario:")
print("   Squeezing: 3.5 dB (BBO achievable)")
print("   LO noise: 0.05 (realistic)")
print("   Electronic noise: -13 dB (typical)")

res_bbo = sim.simulate_complete_protocol(3.5, 0.05, -13)
print(f"\n   Simulated:")
print(f"   • H_min_smooth: {res_bbo['H_min_smooth']:.2f} bits")
print(f"   • Secure bits/sample: {res_bbo['secure_bits_per_sample']:.3f}")
print(f"   • Rate: {res_bbo['rate_mbps']:.1f} Mbps")
print(f"   • Compared to paper: {res_bbo['rate_mbps']/580.7*100:.1f}%")

# Test 3: Sweep squeezing
print("\n3. Sweep squeezing level (LO=0.05, elec=-13):")
print("   dB | Rate (Mbps) | Bits/sample")
print("   ---|-------------|------------")

for db in [0, 1, 2, 3, 4, 5, 6]:
    res = sim.simulate_complete_protocol(db, 0.05, -13, N_measurements=1e6)
    print(f"   {db:2.0f} | {res['rate_mbps']:11.1f} | {res['secure_bits_per_sample']:.3f}")

print("\n" + "="*70)
print("KEY EXPECTATIONS:")
print("="*70)
print("1. 0 dB squeezing → ~0 Mbps (no quantum advantage)")
print("2. 3 dB squeezing → ~200-300 Mbps (BBO realistic)")
print("3. 6 dB squeezing → ~500-600 Mbps (matches paper)")
print("4. Rates should INCREASE with squeezing")
print("5. Rates should DECREASE with noise")
print("\nIf these match, your simulation is CORRECT!")