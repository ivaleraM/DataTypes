function simComp(datasetC,Nits,KK,itt)
    %% runs baseline, which assumes all the continuous variables to be Gaussian
    % and all the dicrete variables to be categorical
    %
    % Inputs:
    %   datasetC: name of the dataset to be inferred
    %   Nit: number of interations of the Gibbs sampler
    %   KK: low rank representation complexity (i.e., number of features)
    %   itt: number of simulation
    % Outputs: returns void but saves a file with the restuls, i.e., the
    % test log-likelihood.
    
    addpath auxFunc
    addpath Baseline
    %Load Observations
    load(['datasets/' datasetC '.mat']);

    ouptutFold= ['resultsComp/' datasetC '/'];
    mkdir(ouptutFold);
    [N D]=size(X);
    
    load(['datasets/' datasetC 'Miss.mat']);
    Xmiss=X;        % Observation matrix
    Xmiss(miss)= -1; % Missing data are coded as missing

    [Kest West countErr LIK]= Baseline(Xmiss,T,R,W,1,1,1,0.001,1,Nits,KK, X);
    if countErr==0
    output=[ouptutFold 'KK' num2str(KK) '_Nsim' num2str(Nits) '_it' num2str(itt) '.mat'];
    save (output, 'LIK');
    end
end