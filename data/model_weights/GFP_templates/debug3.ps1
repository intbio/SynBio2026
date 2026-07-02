function Get-Test1 { return @{ A = 'hello' } }
function Get-Test2 { return ,@{ A = 'hello' } }

$t1 = Get-Test1
$t2 = Get-Test2

Write-Output "Test1 type: $($t1.GetType().FullName)"
Write-Output "Test1 count: $($t1.Count)"
Write-Output "Test1 A: '$($t1['A'])'"

Write-Output "Test2 type: $($t2.GetType().FullName)"
Write-Output "Test2 count: $($t2.Count)"
Write-Output "Test2 A: '$($t2['A'])'"

$chains = Get-Test1
Write-Output "Keys from test1: $($chains.Keys -join ',')"
Write-Output "Value: '$($chains[$chains.Keys[0]])'"
