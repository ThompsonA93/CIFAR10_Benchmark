for($i = 0; $i -lt 1; $i++){
    Write-Host "[$i] Script iteration" 
    python.exe .\src\01_CIFAR10_SKlearn.py
    Write-Host ""
    Write-Host ""
}
Write-Host "Script finished."
Write-Host ""