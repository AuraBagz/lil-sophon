# Start the DuplexJev voice bridge detached (127.0.0.1:8857 only). -Stop stops only this bridge.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File .\start-bridge.ps1 [-Stop] [-SkipGate]
param([switch]$Stop, [switch]$SkipGate)
$root = $PSScriptRoot
$pidFile = "$root\out\bridge\bridge.pid"
$mine = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*duplexjev*bridge.py*' }
if ($Stop) {
  foreach ($p in $mine) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue; "stopped $($p.ProcessId)" }
  Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*duplexjev*phonon_worker.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
  return
}
if ($mine) { "already running: $($mine.ProcessId -join ',')"; return }
$env:PYTHONUNBUFFERED = '1'
$args = @("$root\bridge.py")
if ($SkipGate) { $args += '--skip-gate' }
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$log = "$root\out\bridge\bridge-$stamp.log"
$p = Start-Process -FilePath "$root\venv\Scripts\python.exe" -ArgumentList $args -WorkingDirectory $root `
  -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError "$log.err" -PassThru
$p.Id | Set-Content $pidFile
"started pid $($p.Id), log $log (stderr $log.err); ready in about 15 s at http://127.0.0.1:8857/"
