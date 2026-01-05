
import zipfile
import os
from pathlib import Path

OUTPUT_NAME = "Arrythmia_Project_Enhanced.zip"
ROOT_DIR = Path.cwd()

# Blacklist of directories to EXCLUDE
EXCLUDE_DIRS = {
    '.git', '.gemini', '__pycache__', 'node_modules', 
    'outputs', 'logs', 'venv', 'env', '.idea', '.vscode'
}

# Blacklist of Extensions
EXCLUDE_EXTS = {'.pyc', '.pyd', '.pyo', '.log', '.zip'}

def create_zip():
    print(f"[>] Zipping {ROOT_DIR} into {OUTPUT_NAME}...")
    print(f"    (Excluding: {EXCLUDE_DIRS})")
    
    with zipfile.ZipFile(OUTPUT_NAME, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(ROOT_DIR):
            # Modify dirs in-place to prune traversal
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            
            for file in files:
                if file == OUTPUT_NAME:
                    continue
                
                # Check extension
                if any(file.endswith(ext) for ext in EXCLUDE_EXTS):
                    continue
                    
                file_path = os.path.join(root, file)
                
                # Rel path
                rel_path = os.path.relpath(file_path, ROOT_DIR)
                
                # Skip top-level temp/log/dump files if needed, or specific
                if file == "release_log.txt": continue
                
                # Prefix
                arcname = os.path.join("Arrythmia_Project_v3_Refined", rel_path)
                
                try:
                    zipf.write(file_path, arcname)
                except PermissionError:
                    print(f"    [!] Skip locked: {file}")

    print(f"[+] Successfully created {OUTPUT_NAME}")

if __name__ == "__main__":
    create_zip()
