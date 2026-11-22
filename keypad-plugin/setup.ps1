param([string]$Dotnet = "dotnet")

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\pythonw.exe"
$entry = Join-Path $projectRoot "paper_workbench.py"
if (!(Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Set up the project with uv sync first."
}

$env:DOTNET_CLI_TELEMETRY_OPTOUT = "1"
$portable = Join-Path $env:LOCALAPPDATA "LogiInsightTools\dotnet\dotnet.exe"
if ($Dotnet -eq "dotnet" -and !(& dotnet --list-sdks 2>$null) -and (Test-Path -LiteralPath $portable)) {
    $Dotnet = $portable
}
$project = Join-Path $PSScriptRoot "src\PaperDeskPlugin.csproj"
& $Dotnet build $project --configuration Release --nologo
if ($LASTEXITCODE -ne 0) { throw "The Logi Actions plugin failed to build." }

$data = Join-Path $env:LOCALAPPDATA "LogiInsightPaper"
$plugins = Join-Path $env:LOCALAPPDATA "Logi\LogiPluginService\Plugins"
$null = New-Item -ItemType Directory -Force $data, $plugins
$launcher = @{ python = $python; script = $entry } | ConvertTo-Json -Compress
Set-Content -LiteralPath (Join-Path $data "launcher.json") -Value $launcher -Encoding utf8
$package = (Resolve-Path (Join-Path $PSScriptRoot "bin\Release")).Path

# Icon-only templates for the app folder actions: one full-screen image item and no title item.
$templates = Join-Path $package "icontemplates"
$null = New-Item -ItemType Directory -Force $templates
$folders = @{ VsCode = "vscode"; Matlab = "matlab"; Latex = "latex"; ChatGpt = "chatgpt"; KiCad = "kicad"; LtSpice = "ltspice"; Git = "git"; Plan = "plan" }
# The nine paper keys draw their own image for the current menu. A template for them would
# replace that image with the template's empty one, so they get none.
Remove-Item -LiteralPath (Join-Path $templates "Loupedeck.PaperDeskPlugin.PaperKey.ict") -ErrorAction SilentlyContinue
$actions = @{}
foreach ($name in $folders.Keys) {
    $icon = Join-Path $PSScriptRoot "icons\$($folders[$name]).png"
    $actions["Loupedeck.PaperDeskPlugin.${name}PaperFolder"] = [Convert]::ToBase64String([IO.File]::ReadAllBytes($icon))
}
foreach ($action in $actions.Keys) {
    $template = [ordered]@{
        backgroundColor = 4279770148
        items = @([ordered]@{
            '$type' = "Loupedeck.Service.ActionIconImageItem, LoupedeckShared"
            image = $actions[$action]; imageFileName = $null; imageColor = 4294967295
            imageRotation = "None"; isVisible = $true; itemType = "Image"
            area = [ordered]@{ x = 0; y = 0; width = 100; height = 100; isFullScreen = $true }
        })
    }
    $json = $template | ConvertTo-Json -Depth 5
    [IO.File]::WriteAllText((Join-Path $templates "$action.ict"), $json)
}

# A service that was stopped while loading leaves a crash marker that disables the plugin on the next start.
Remove-Item -LiteralPath (Join-Path $env:LOCALAPPDATA "Logi\LogiPluginService\Logs\plugin_crashes\PaperDeskPlugin.dll") -ErrorAction SilentlyContinue

$link = Join-Path $plugins "PaperDeskPlugin.link"
Set-Content -LiteralPath $link -Value "$package\" -Encoding utf8
Write-Output "Registered LogiInsight Paper with Logi Plugin Service."
