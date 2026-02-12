import sys
import subprocess
import config
from pathlib import Path

# List of scripts in execution order
SCRIPTS = [
    "01_limpieza_inicial.py",
    "02_feature_engineering.py",
    "03_modelos.py",
    "09_analisis_municipal.py",
    "10_analisis_avanzado.py",
    "11_modelos_ensemble.py",
    "12_validacion_cierre_2025.py",
    "08_reporte_final.py"
]

def run_pipeline():
    print("🚀 STARTED RENTAS CEDIDAS PIPELINE")
    print(f"📂 Base Dir: {config.BASE_DIR}")
    
    for script in SCRIPTS:
        print(f"\n" + "="*50)
        print(f"▶️ Running {script}...")
        print("="*50)
        
        script_path = config.BASE_DIR / "scripts" / script
        
        if not script_path.exists():
            print(f"❌ Script not found: {script_path}")
            sys.exit(1)
            
        # Run script as a separate process
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)], 
                text=True, 
                check=False # We handle returncode manually
            )
            
            if result.returncode != 0:
                print(f"\n❌ Execution FAILED for {script}. Return Code: {result.returncode}")
                sys.exit(1)
            else:
                print(f"\n✅ Finished {script} successfully.")
                
        except Exception as e:
            print(f"❌ Error trying to run {script}: {e}")
            sys.exit(1)
            
    print("\n" + "="*50)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*50)

if __name__ == "__main__":
    run_pipeline()
