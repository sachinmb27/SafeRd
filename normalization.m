function [Z, m, s] = featureNormalize(Z)
    Z_norm = Z;
    mu = zeros(1, size(Z, 2));
    sigma = zeros(1, size(Z, 2));
    m = mean(Z);
    s =std(Z);
    Z_norm = (Z - m) ./ s ;
end