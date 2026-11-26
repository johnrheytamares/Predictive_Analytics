$ErrorActionPreference = 'Stop'
$base = 'http://127.0.0.1:8013'
$key = 'local-v32-smoke-only-key'
$outDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$report = [ordered]@{
  suite = 'docker_v32_smoke'
  image = 'reservision-predictive:3.2-prepared'
  container = 'reservision-v32-smoke-test'
  port = 8013
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

try {
  docker rm -f reservision-v32-smoke-test 2>$null | Out-Null
  docker build -f Dockerfile.v32 -t reservision-predictive:3.2-prepared .
  docker run -d --name reservision-v32-smoke-test -p 8013:8002 `
    -e PORT=8002 `
    -e PREDICTION_API_KEY=$key `
    reservision-predictive:3.2-prepared | Out-Null

  $ready = $false
  for ($i = 0; $i -lt 40; $i++) {
    try {
      $probe = Get-Json "$base/health"
      if ($probe.code -eq 200 -and $probe.json.model_ready -eq $true) { $ready = $true; break }
    } catch {}
    Start-Sleep -Seconds 2
  }
  Add-Check 'container_ready' $ready 'health model_ready within timeout'

  $healthResp = Get-Json "$base/health"
  $health = $healthResp.json
  Add-Check 'version_3_2' ($health.metadata.model_version -eq '3.2') ([string]$health.metadata.model_version)
  Add-Check 'production_approved' ($health.production_approved -eq $true -and $health.metadata.production_approved -eq $true) ("approved=$($health.production_approved)")
  Add-Check 'champions' (
    $health.metadata.targets.bookings.selected_model -eq 'two_stage' -and
    $health.metadata.targets.guests.selected_model -eq 'guests_per_booking_sumratio' -and
    $health.metadata.postprocess.selected_mode -eq 'floor_round'
  ) ("b=$($health.metadata.targets.bookings.selected_model); g=$($health.metadata.targets.guests.selected_model)")

  $bm = [math]::Round($health.weekly_performance.bookings.operational_floor_round.mae, 2)
  $br = [math]::Round($health.weekly_performance.bookings.operational_floor_round.rmse, 2)
  $gm = [math]::Round($health.weekly_performance.guests.operational_floor_round.mae, 2)
  $gr = [math]::Round($health.weekly_performance.guests.operational_floor_round.rmse, 2)
  Add-Check 'booking_mae_rmse' ($bm -eq 2.54 -and $br -eq 3.87) ("$bm/$br")
  Add-Check 'guest_mae_rmse' ($gm -eq 29.18 -and $gr -eq 40.94) ("$gm/$gr")

  $headers = @{ 'X-API-Key' = $key }
  $sample = Get-Json "$base/forecast?start_date=2027-01-01&end_date=2027-02-08" $headers
  $bookings = ($sample.json.forecast | Measure-Object -Property predicted_bookings -Sum).Sum
  $guests = ($sample.json.forecast | Measure-Object -Property predicted_total_guests -Sum).Sum
  Add-Check 'sample_12_108' ($sample.code -eq 200 -and $bookings -eq 12 -and $guests -eq 108) ("http=$($sample.code); b=$bookings; g=$guests")

  $report.passed = $true
}
catch {
  $report.error = "$_"
  $report.passed = $false
}
finally {
  docker rm -f reservision-v32-smoke-test 2>$null | Out-Null
  ($report | ConvertTo-Json -Depth 8) | Set-Content -Path (Join-Path $outDir 'docker_smoke_results.json') -Encoding utf8
  if (-not $report.passed) { exit 1 }
  Write-Host 'docker_v32_smoke PASSED'
}
