# -*- coding: utf-8 -*-
import os
import subprocess
import time

print("Step 1: Killing any stuck latex/perl/biber processes...")
subprocess.run("taskkill /f /im perl.exe 2>nul", shell=True)
subprocess.run("taskkill /f /im xelatex.exe 2>nul", shell=True)
subprocess.run("taskkill /f /im biber.exe 2>nul", shell=True)
subprocess.run("taskkill /f /im bibtex.exe 2>nul", shell=True)

# Delete auxiliary files manually to ensure no locks or stale formats
print("Step 2: Cleaning auxiliary files manually...")
exts = [".aux", ".log", ".synctex.gz", ".fls", ".bbl", ".bcf", ".run.xml", ".toc", ".out", ".blg", ".xdv", ".idx", ".ilg", ".ind"]
for ext in exts:
    fpath = "main" + ext
    if os.path.exists(fpath):
        try:
            os.remove(fpath)
            print(f"Removed: {fpath}")
        except Exception as e:
            print(f"Warning: Could not remove {fpath}: {e}")

print("Step 3: Compiling main.tex via XeLaTeX...")
# Run latexmk with full output so we can see it
cmd = ["latexmk", "-xelatex", "-interaction=nonstopmode", "-file-line-error", "main.tex"]
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

start_time = time.time()
while True:
    line = process.stdout.readline()
    if not line:
        break
    # Print the compilation progress in real-time!
    print(line, end="")
    
    # Safety timeout of 120 seconds
    if time.time() - start_time > 120:
        print("\nTimeout: Compilation took longer than 120 seconds, killing...")
        process.terminate()
        break

process.wait()
print(f"Step 4: Compilation finished with Exit Code: {process.returncode}")
if os.path.exists("main.pdf"):
    print(f"SUCCESS: main.pdf built successfully! Size: {os.path.getsize('main.pdf')} bytes")
else:
    print("FAILED: main.pdf was not built.")
