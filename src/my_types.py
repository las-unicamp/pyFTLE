from nptyping import Float32, NDArray, Shape

ArrayFloat32Nx2 = NDArray[Shape["*, 2"], Float32]
ArrayFloat32Nx3 = NDArray[Shape["*, 3"], Float32]
ArrayFloat32NxN = NDArray[Shape["*, *"], Float32]
