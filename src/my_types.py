from nptyping import Float64, NDArray, Shape

ArrayFloat64N = NDArray[Shape["*"], Float64]
ArrayFloat64MxN = NDArray[Shape["*, *"], Float64]
ArrayFloat64Nx2 = NDArray[Shape["*, 2"], Float64]
ArrayFloat64Nx2x2 = NDArray[Shape["*, 2, 2"], Float64]
ArrayFloat64N4x2 = NDArray[Shape["*, 4, 2"], Float64]
