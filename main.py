import os
import subprocess

def run_script(script_name):
    print(f"\n{'='*50}\nRunning {script_name}...\n{'='*50}")
    # Using the virtual environment's python
    python_exec = os.path.join("venv", "Scripts", "python.exe") if os.name == 'nt' else os.path.join("venv", "bin", "python")
    result = subprocess.run([python_exec, script_name], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error in {script_name}:\n{result.stderr}")
        exit(1)

if __name__ == "__main__":
    scripts = [
        "src/data_loader.py",
        "src/eda.py",
        "src/preprocessing.py",
        "src/model_trainer.py",
        "src/explainer.py"
    ]
    
    for script in scripts:
        run_script(script)
        
    print("\nAll pipeline steps completed successfully!")
    print("You can now run the Streamlit dashboard using: streamlit run app/app.py")
