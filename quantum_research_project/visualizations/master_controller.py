"""
Master controller for all visualizations
"""
import subprocess
import sys
import os

def run_visualization(choice):
    """Run selected visualization"""
    
    visualizations = {
        '1': 'animation_simulation.py',
        '2': 'step_by_step_demo.py',
        '3': 'live_quantum_dashboard.py',
        '4': 'all'  # Run all sequentially
    }
    
    if choice == '4':
        print("🎬 Running ALL visualizations sequentially...")
        for key in ['1', '2', '3']:
            print(f"\n{'='*50}")
            print(f"Running: {visualizations[key]}")
            print(f"{'='*50}")
            subprocess.run([sys.executable, visualizations[key]])
            input("\nPress Enter to continue to next visualization...")
    elif choice in visualizations:
        print(f"🚀 Running {visualizations[choice]}...")
        subprocess.run([sys.executable, visualizations[choice]])
    else:
        print("❌ Invalid choice!")

def main():
    """Main menu"""
    print("\n" + "="*60)
    print("🌌 BBO SQUEEZED LIGHT QRNG VISUALIZATION SUITE")
    print("="*60)
    print("\nChoose visualization:")
    print("1. 🎬 Full Animation (Best for presentations)")
    print("2. 📚 Step-by-Step Demo (Best for learning)")
    print("3. 📊 Live Dashboard (Best for monitoring)")
    print("4. 🚀 Run ALL visualizations")
    print("0. ❌ Exit")
    
    while True:
        choice = input("\nEnter choice (0-4): ").strip()
        
        if choice == '0':
            print("👋 Goodbye!")
            break
        elif choice in ['1', '2', '3', '4']:
            run_visualization(choice)
        else:
            print("❌ Please enter 0-4")

if __name__ == "__main__":
    # Check all files exist
    required_files = ['animation_simulation.py', 
                     'step_by_step_demo.py', 
                     'live_quantum_dashboard.py']
    
    missing = [f for f in required_files if not os.path.exists(f)]
    
    if missing:
        print(f"❌ Missing files: {missing}")
        print("Please create all three Python files first!")
    else:
        main()