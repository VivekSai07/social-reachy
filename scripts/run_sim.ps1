# Launches the Reachy Mini daemon in simulation mode (MuJoCo viewer).
# Usage: .\scripts\run_sim.ps1 [-Scene empty|minimal]
param(
    [ValidateSet("empty", "minimal")]
    [string]$Scene = "empty"
)

& "$PSScriptRoot\..\.venv\Scripts\reachy-mini-daemon.exe" --sim --scene $Scene
