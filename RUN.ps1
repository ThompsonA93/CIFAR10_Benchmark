for($i = 0; $i -lt 3; $i++){
    Write-Host "[$i] Script iteration" 
    #python.exe .\src\01_CIFAR10_SKlearn.py
    python.exe .\src\02_CIFAR10_Keras.py
    Write-Host ""
    Write-Host ""
}
Write-Host "Script finished."
Write-Host ""