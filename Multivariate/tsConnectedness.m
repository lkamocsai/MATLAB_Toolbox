function tblConnactedness = tsConnectedness(FEVDGen)
% ------------------------------------------------------------------------------------
% Function to estimate spillovers between variables
% ------------------------------------------------------------------------------------
% INPUT: FEVD: Forecast Error Variance Decompositin (K x K x h)
% ------------------------------------------------------------------------------------
% OUTPUT: tblConnactedness: Spillover table (K x K x h)        
% ------------------------------------------------------------------------------------
% Refrences:
% 1. L.Kilian and H.Lütkepohl - Structural Vector Autoregressive Analysis (Oxford, 2017)
% 2. H.Lütkepohl - New Introduction to Multiple Time Series Analysis (Springer, 2005)
% 3. S.Hurn, V.Martin, D.Harris - Econometric Modelling with Time Series (Cambridge, 2012)
% 4. F.X.Diebold, K.Yilmaz - Financial and Macroeconomic Connectedness (Oxford, 2015)
% 5. H.M.Pesaran - Time Series and Panel Data Econometrics (Oxford, 2015)
% ------------------------------------------------------------------------------------
%
%----------------------------------(1) check inputs ----------------------------------

arguments
    FEVDGen {mustBeNonempty,mustBeNumeric}
end

%----------------------------------(2) set env ---------------------------------------
[~,K,h] = size(FEVDGen);

% Set working table
normFEVDGen = FEVDGen;

%-----------------------------(3) spillovers calculation -----------------------------
for hh = 1:h
    % Normalize data as Diebold-Yilmaz recommended
    for ii = 1:K
        normFEVDGen(ii,:,hh) = (FEVDGen(ii,:,hh)./sum(FEVDGen(ii,:,hh),2))*100;
    end
    % total volatility spillover (system wide connectedness)
    Stotal(:,:,hh) = sum(sum(normFEVDGen(:,:,hh),2) - diag(normFEVDGen(:,:,hh)))/K;
    % spillovers received by variable i from ALL other variables j (total directional connectedness from others to i)
    Sidot(:,:,hh) = sum(normFEVDGen(:,:,hh),2) - diag(normFEVDGen(:,:,hh));
    % spillovers transmitted by variable i to ALL other variables j (total directional connectedness from i to others)
    Sdoti(:,:,hh) = sum(normFEVDGen(:,:,hh),1) - diag(normFEVDGen(:,:,hh))';
    % net spillover from variable i to all other variables j (net total directional effect)
    Si(:,:,hh) = Sdoti(:,:,hh)' - Sidot(:,:,hh);
    % Net pairwise spillover
    for ii = 1:K
        for jj = 1:K
            Sij(ii,jj,hh) = (normFEVDGen(jj,ii,hh) - normFEVDGen(ii,jj,hh));
        end
    end

    % Save results
    tblConnactedness(:,:,hh) = [normFEVDGen(:,:,hh) Sidot(:,:,hh); [Sdoti(:,:,hh) Stotal(:,:,hh)]];
end

end