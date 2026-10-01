import os
import glob
import subprocess

files = glob.glob(r'd:\PROJECTS\harmonyshield\harmonyshield\data\external_demo\*.mp3')
print(f"Found {len(files)} files")

for f in files:
    print("\n" + "="*80)
    print("Running inference on file...")
    print("="*80)
    subprocess.run([r'd:\PROJECTS\harmonyshield\harmonyshield\venv\Scripts\python.exe', r'd:\PROJECTS\harmonyshield\harmonyshield\scripts\demo_inference.py', '--audio', f])
