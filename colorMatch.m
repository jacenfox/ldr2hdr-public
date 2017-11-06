function imgCorrected = colorMatch(imgMark3, T)
[nrows, ncols, nbands] = size(imgMark3);
imgCorrected = reshape((reshape(imgMark3, nrows*ncols, nbands)*T), ...
        nrows, ncols, nbands);
imgCorrected = max(imgCorrected, 0);
