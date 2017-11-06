function [T, scale] = rescaleColorMat(T, CanonEVnew_calib, RicohEV_calib, CanonEV, RicohEV)

CanonExposure = 2.^(-(CanonEV-CanonEVnew_calib)); % canon * exposure = canon_calib
RicohExposure = 2.^(-(RicohEV-RicohEV_calib));
scale = RicohExposure / CanonExposure; % newT = scale * T
T = scale * T;