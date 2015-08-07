function [projected_thickness] = test_xray_tie (I2, IinVal, Mag, R2, mu, delta, ps, reg)
[M, N] = size(I2);
log_output = log(I2);
projected_thickness = -(1/mu) * log_output;
end