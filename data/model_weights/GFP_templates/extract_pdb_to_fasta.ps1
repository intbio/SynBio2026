# Extract protein sequences from PDB files into a single FASTA file.
# Uses SEQRES records when present, otherwise falls back to ATOM CA records.

$AA3 = @{
    'ALA'='A'; 'ARG'='R'; 'ASN'='N'; 'ASP'='D'; 'CYS'='C'; 'GLN'='Q'; 'GLU'='E'; 'GLY'='G'
    'HIS'='H'; 'ILE'='I'; 'LEU'='L'; 'LYS'='K'; 'MET'='M'; 'PHE'='F'; 'PRO'='P'; 'SER'='S'
    'THR'='T'; 'TRP'='W'; 'TYR'='Y'; 'VAL'='V'; 'SEC'='U'; 'PYL'='O'
    'MSE'='M'; 'ASX'='B'; 'GLX'='Z'; 'XAA'='X'; 'UNK'='X'
    'CRO'='X'; 'CRQ'='X'
}

function Convert-Residue {
    param([string]$ResName)
    $key = $ResName.ToUpper()
    if ($AA3.ContainsKey($key)) { return $AA3[$key] }
    return 'X'
}

function Get-PdbSequences {
    param([string]$PdbPath)

    $chainsSeqres = @{}
    $chainsAtom = @{}

    foreach ($line in [System.IO.File]::ReadLines($PdbPath)) {
        if ($line.StartsWith('SEQRES')) {
            $chain = $line.Substring(11, 1).Trim()
            $resPart = $line.Substring(19).Trim()
            $resNames = $resPart -split '\s+' | Where-Object { $_ -match '^[A-Z]{3}$' }
            if (-not $chainsSeqres.ContainsKey($chain)) {
                $chainsSeqres[$chain] = [System.Collections.Generic.List[string]]::new()
            }
            foreach ($r in $resNames) { $chainsSeqres[$chain].Add($r) }
        }
        elseif ($line.StartsWith('ATOM')) {
            $atomName = $line.Substring(12, 4).Trim()
            if ($atomName -ne 'CA') { continue }
            $resName = $line.Substring(17, 3).Trim()
            $chain = $line.Substring(21, 1).Trim()
            $resNum = $line.Substring(22, 4).Trim()
            $altLoc = $line.Substring(16, 1)
            if ($altLoc -ne ' ' -and $altLoc -ne 'A') { continue }
            if (-not $chainsAtom.ContainsKey($chain)) {
                $chainsAtom[$chain] = [ordered]@{}
            }
            $key = "$resNum"
            if (-not $chainsAtom[$chain].Contains($key)) {
                $chainsAtom[$chain][$key] = $resName
            }
        }
    }

    $result = @{}
    if ($chainsSeqres.Count -gt 0) {
        foreach ($chain in $chainsSeqres.Keys) {
            $seq = ($chainsSeqres[$chain] | ForEach-Object { Convert-Residue $_ }) -join ''
            $result[$chain] = $seq
        }
    }
    else {
        foreach ($chain in $chainsAtom.Keys) {
            $seq = ($chainsAtom[$chain].Values | ForEach-Object { Convert-Residue $_ }) -join ''
            $result[$chain] = $seq
        }
    }

    # Comma prevents PowerShell from flattening single-entry hashtables on return.
    return ,$result
}

function Get-SequenceName {
    param([string]$FileName)
    $base = [System.IO.Path]::GetFileNameWithoutExtension($FileName)
    if ($base -match '^(.+)_[A-Za-z0-9]{4}$') {
        return $Matches[1]
    }
    return $base
}

$pdbDir = $PSScriptRoot
$outputFile = Join-Path $pdbDir 'GFPs.fasta'
$pdbFiles = Get-ChildItem -Path $pdbDir -Filter '*.pdb' | Sort-Object Name

$fastaLines = [System.Collections.Generic.List[string]]::new()
$warnings = [System.Collections.Generic.List[string]]::new()

foreach ($pdb in $pdbFiles) {
    $chains = Get-PdbSequences -PdbPath $pdb.FullName
    if ($chains.Count -eq 0) {
        $warnings.Add("$($pdb.Name): no sequence found")
        continue
    }

    $sortedChains = @($chains.Keys | Sort-Object)
    $firstChain = $sortedChains[0]
    $firstSeq = $chains[$firstChain]

    if ($chains.Count -gt 1) {
        $allSame = $true
        foreach ($chain in $sortedChains) {
            if ($chains[$chain] -ne $firstSeq) {
                $allSame = $false
                $warnings.Add("$($pdb.Name): chains differ (chain $firstChain len=$($firstSeq.Length) vs chain $chain len=$($chains[$chain].Length))")
            }
        }
        if ($allSame) {
            Write-Host "$($pdb.Name): $($chains.Count) identical chains, using chain $firstChain"
        }
    }

    $seqName = Get-SequenceName -FileName $pdb.Name
    $fastaLines.Add(">$seqName")

    for ($i = 0; $i -lt $firstSeq.Length; $i += 80) {
        $len = [Math]::Min(80, $firstSeq.Length - $i)
        $fastaLines.Add($firstSeq.Substring($i, $len))
    }

    Write-Host "$($pdb.Name) -> >$seqName ($($firstSeq.Length) aa, chain $firstChain)"
}

[System.IO.File]::WriteAllLines($outputFile, $fastaLines)

Write-Host ""
Write-Host "Saved $($pdbFiles.Count) sequences to $outputFile"

if ($warnings.Count -gt 0) {
    Write-Host ""
    Write-Host "Warnings:"
    foreach ($w in $warnings) { Write-Host "  $w" }
}
