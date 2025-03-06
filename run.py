"""Run script for the Streamlit application."""
import subprocess
import sys
from pathlib import Path
from src.core.constants import DirectoryPath


def main():
    """Run the Streamlit application."""
    # fil_path = GetPath.ROOT_DIR.value / "app"/ "main.py"

    
    app_path = Path(DirectoryPath.STREAMLIT_APP.value)

   
    if not app_path.exists():
        print(f"Error: Could not find {app_path}")
        sys.exit(1)
        
    try:
        subprocess.run(["streamlit", "run", str(app_path)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 