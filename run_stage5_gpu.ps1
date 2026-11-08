param(
    [string]$PythonBin = "D:\Anaconda\envs\papaer_tep\python.exe",
    [string]$ConfigPath = "$PSScriptRoot\configs\stage5.yaml"
)

$ErrorActionPreference = "Stop"

function Invoke-Stage5Python {
    param([string[]]$Arguments)
    & $PythonBin @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Stage-5 command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

Invoke-Stage5Python @("$PSScriptRoot\scripts\run_protocol.py", "--config", $ConfigPath, "--protocol", "protocol_1")
Invoke-Stage5Python @("$PSScriptRoot\scripts\run_protocol.py", "--config", $ConfigPath, "--protocol", "protocol_2")
Invoke-Stage5Python @("-m", "eval.summarize", "--config", $ConfigPath)
Invoke-Stage5Python @("-m", "eval.visualize", "--config", $ConfigPath, "--protocol", "protocol_1", "--seed", "0")
Invoke-Stage5Python @("-m", "eval.visualize", "--config", $ConfigPath, "--protocol", "protocol_2", "--seed", "0")

# Design note (paper Sec. 5.2--5.8): this Windows entry runs the 40-run
# conference matrix, summaries, and figures using the centralized YAML config.
