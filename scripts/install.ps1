# One-click laptop install (Windows PowerShell).
python -m venv .venv
.\.venv\Scripts\pip install -e ".[dev]"
Write-Host "Done. Run: .\.venv\Scripts\hydra-sim.exe"
