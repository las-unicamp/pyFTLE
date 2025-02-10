from nptyping import Float32, NDArray, Shape

ArrayFloat32N = NDArray[Shape["*"], Float32]
ArrayFloat32Nx2 = NDArray[Shape["*, 2"], Float32]
ArrayFloat32Nx2x2 = NDArray[Shape["*, 2, 2"], Float32]
ArrayFloat32N4x2 = NDArray[Shape["*, 4, 2"], Float32]
