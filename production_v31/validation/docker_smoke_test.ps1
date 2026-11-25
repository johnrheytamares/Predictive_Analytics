$ErrorActionPreference = 'Stop'
$base = 'http://127.0.0.1:8012'
$key = 'local-smoke-only-key'
$outDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$report = [ordered]@{
  suite = 'docker_v31_smoke'
  image = 'reservision-predictive:3.1-prepared'
  container = 'reservision-v31-smoke-test'
  port = 8012
  passed = $false
  checks = @()
}

function Add-Check([string]$name, [bool]$ok, [string]$detail) {
  $script:report.checks += [ordered]@{ name = $name; ok = $ok; detail = $detail }
  if (-not $ok) { throw "CHECK_FAILED: $name :: $detail" }
}

function Get-Json([string]$url, [hashtable]$headers = $null) {
  $args = @('-sS', '-w', '\n%{http_code}', $url)
  if ($headers) {
    foreach ($k in $headers.Keys) { $args = @('-H', ("{0}: {1}" -f $k, $headers[$k])) + $args }
  }
  $raw = & curl.exe @args
  $lines = $raw -split "`n"
  $code = [int]$lines[-1]
  $body = ($lines[0..($lines.Length - 2)] -join "`n").Trim()
  return @{ code = $code; body = $body; json = ($(if ($body) { $body | ConvertFrom-Json } else { $null })) }
}

$ls = docker exec reservision-v31-smoke-test ls -1 `
  /app/production_v31/api.py `
  /app/production_v31/serve.py `
  /app/experiments/reserVision_v3_1_two_stage_rc_20261124/artifacts/forecast_models_metadata.json `
  /app/api.py `
  /app/bookings_model.joblib
Add-Check 'artifacts_present' (($ls -join "`n") -match 'production_v31/api.py' -and ($ls -join "`n") -match 'forecast_models_metadata.json') ($ls -join "`n")

$envDump = docker exec reservision-v31-smoke-test printenv
Add-Check 'no_shadow_env' ((($envDump -join "`n") -notmatch 'RESERVISION_SHADOW|SHADOW_MODE')) ($envDump -join "`n")
Add-Check 'artifact_dir_set' (($envDump -join "`n") -match 'PREDICTION_ARTIFACT_DIR=/app/experiments/reserVision_v3_1_two_stage_rc_20261124/artifacts') ($envDump -join "`n")
Add-Check 'cmd_production_serve' ((docker inspect -f '{{json .Config.Cmd}}' reservision-v31-smoke-test) -match 'production_v31.serve') (docker inspect -f '{{json .Config.Cmd}}' reservision-v31-smoke-test)

$healthResp = Get-Json "$base/health"
$health = $healthResp.json
Add-Check 'health_ok' ($healthResp.code -eq 200 -and $health.status -eq 'ok' -and $health.model_ready -eq $true) ("http=$($healthResp.code); status=$($health.status); ready=$($health.model_ready)")
Add-Check 'version_3_1' ($health.metadata.model_version -eq '3.1') ([string]$health.metadata.model_version)
Add-Check 'champions' (
  $health.metadata.targets.bookings.selected_model -eq 'two_stage' -and
  $health.metadata.targets.guests.selected_model -eq 'robust_median_guests' -and
  $health.metadata.postprocess.selected_mode -eq 'floor_round'
) ("b=$($health.metadata.targets.bookings.selected_model); g=$($health.metadata.targets.guests.selected_model); p=$($health.metadata.postprocess.selected_mode)")
Add-Check 'production_prepared_not_shadow' (
  $health.metadata.serving_mode -eq 'production_prepared' -and
  $health.metadata.shadow_serving_only -eq $false -and
  $health.production_approved -eq $true -and
  $health.lifecycle.prospective_accuracy_validated -eq $false
) ("mode=$($health.metadata.serving_mode); shadow=$($health.metadata.shadow_serving_only); approved=$($health.production_approved)")
Add-Check 'history_refresh_lifecycle' (
  $health.lifecycle.history_refresh_supported -eq $true -and
  $health.lifecycle.dataset_sync_refreshes_served_history -eq $false -and
  $health.lifecycle.model_fitted -eq $false -and
  $health.lifecycle.evaluation_metrics_refreshed -eq $false -and
  $health.lifecycle.automatic_v3_fallback -eq $false
) ("refresh=$($health.lifecycle.history_refresh_supported); sync_refreshes=$($health.lifecycle.dataset_sync_refreshes_served_history)")

$bw = [math]::Round($health.weekly_performance.bookings.operational_floor_round.wape, 2)
$gw = [math]::Round($health.weekly_performance.guests.operational_floor_round.wape, 2)
Add-Check 'weekly_metrics' ($bw -eq 29.96 -and $gw -eq 38.79) ("bookings=$bw guests=$gw")
$report.weekly_wape = @{ bookings = $bw; guests = $gw }

$miss = Get-Json "$base/forecast?start_date=2026-11-25&end_date=2026-12-01"
$wrong = Get-Json "$base/forecast?start_date=2026-11-25&end_date=2026-12-01" @{ 'X-API-Key' = 'wrong-key' }
Add-Check 'auth_missing_401' ($miss.code -eq 401) "status=$($miss.code)"
Add-Check 'auth_wrong_401' ($wrong.code -eq 401) "status=$($wrong.code)"
Add-Check 'health_public' ($healthResp.code -eq 200) "status=$($healthResp.code)"

$forecasts = @{}
foreach ($days in 7, 30, 90) {
  $start = '2026-11-25'
  $end = ([datetime]$start).AddDays($days - 1).ToString('yyyy-MM-dd')
  $resp = Get-Json "$base/forecast?start_date=$start&end_date=$end" @{ 'X-API-Key' = $key }
  $f = $resp.json
  $ok = ($resp.code -eq 200) -and ($f.success -eq $true) -and ($f.model_version -eq '3.1') -and ($f.forecast.Count -eq $days) -and
    ($f.model.targets.bookings.selected_model -eq 'two_stage') -and
    ($f.model.targets.guests.selected_model -eq 'robust_median_guests') -and
    ($f.model.postprocess.selected_mode -eq 'floor_round') -and
    ($f.model.serving_mode -eq 'production_prepared') -and
    ($f.model.shadow_serving_only -eq $false) -and
    ($null -ne $f.weekly_performance.bookings.operational_floor_round.wape)
  $fbw = [math]::Round($f.weekly_performance.bookings.operational_floor_round.wape, 2)
  $fgw = [math]::Round($f.weekly_performance.guests.operational_floor_round.wape, 2)
  Add-Check "forecast_${days}d" ([bool]$ok) ("http=$($resp.code); rows=$($f.forecast.Count); bw=$fbw; gw=$fgw; first=$($f.forecast[0].predicted_bookings)/$($f.forecast[0].predicted_total_guests)")
  $forecasts["$days"] = @{
    rows = $f.forecast.Count
    bookings_wape = $fbw
    guests_wape = $fgw
    total_bookings = ($f.forecast | Measure-Object predicted_bookings -Sum).Sum
    total_guests = ($f.forecast | Measure-Object predicted_total_guests -Sum).Sum
  }
}
$report.forecasts = $forecasts

$invalidPayload = Join-Path $env:TEMP 'reservision_v31_invalid_refresh.json'
Set-Content -Path $invalidPayload -Value '{"rows":[{"demand_date":"2026-01-01","bookings":1,"guests":1}]}' -Encoding ascii -NoNewline
$invalidRefresh = & curl.exe -sS -w "`n%{http_code}" -X POST "$base/history/refresh" `
  -H "X-API-Key: $key" -H "Content-Type: application/json" --data-binary "@$invalidPayload"
$invalidLines = @($invalidRefresh -split "`n")
$invalidCode = [int]$invalidLines[-1]
$invalidRaw = if ($invalidLines.Length -gt 1) { ($invalidLines[0..($invalidLines.Length - 2)] -join "`n").Trim() } else { '' }
$invalidBody = if ($invalidRaw) { $invalidRaw | ConvertFrom-Json } else { $null }
Add-Check 'history_refresh_invalid_422' ($invalidCode -eq 422 -and $invalidBody.code -eq 'INVALID_HISTORY') ("http=$invalidCode; code=$($invalidBody.code)")

$manifestJson = docker exec reservision-v31-smoke-test cat /app/production_v31/release_manifest.json | Out-String
$manifestObj = $manifestJson | ConvertFrom-Json
$manifestCount = @($manifestObj.files.PSObject.Properties).Count
Add-Check 'release_manifest_files' ($manifestCount -ge 40) "files=$manifestCount"
Add-Check 'history_refresh_in_manifest' ($null -ne $manifestObj.files.'production_v31/history_refresh.py') 'history_refresh.py hashed'

$report.passed = (($report.checks | Where-Object { -not $_.ok }).Count -eq 0)
($report | ConvertTo-Json -Depth 8) | Set-Content -Path (Join-Path $outDir 'docker_smoke_results.json') -Encoding utf8
Write-Output "SMOKE_PASSED=$($report.passed)"
$report.checks | ForEach-Object {
  $d = [string]$_.detail
  if ($d.Length -gt 160) { $d = $d.Substring(0, 160) }
  Write-Output ("{0} {1} :: {2}" -f $_.ok, $_.name, $d)
}
