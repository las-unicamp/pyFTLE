from nptyping import Float64, NDArray, Shape

ArrayN = NDArray[Shape["*"], Float64]
Array2xN = NDArray[Shape["2, *"], Float64]
Array3xN = NDArray[Shape["3, *"], Float64]
Array4Nx2 = NDArray[Shape["*, 2"], Float64]
Array6Nx3 = NDArray[Shape["*, 3"], Float64]
ArrayNx2 = NDArray[Shape["*, 2"], Float64]
ArrayNx3 = NDArray[Shape["*, 3"], Float64]
ArrayNx2x2 = NDArray[Shape["*, 2, 2"], Float64]
ArrayNx3x3 = NDArray[Shape["*, 3, 3"], Float64]
