function simLik(datasetC,Nits,KK,it)
    %% runs proposed Bayesian method to infer the datatypes in a dataset.
    % Inputs:
    %   datasetC: name of the dataset to be inferred
    %   Nit: number of interations of the Gibbs sampler
    %   KK: low rank representation complexity (i.e., number of features)
    %   itt: number of simulation
    % Outputs: returns void but saves a file with the restuls, i.e., the
    % test log-likelihood adn a vector with the inferred weights for the
    % different datatypes in each dimension.
    addpath auxFunc
    addpath Ccode
    %Load Observations
    load(['datasets/' datasetC '.mat']);

    ouptutFold= ['LIKresults/' datasetC '/']; %loads the dataset
    mkdir(ouptutFold);
    [N D]=size(X);
    
    load(['datasets/' datasetC 'Miss.mat']); % loads the indeces for the 
    % missing values, i.e., the values for which it will compute the
    % log-likelihood
    
    Xmiss=X;        % Observation matrix
    Xmiss(miss)= -1; % Missing data are coded as missing

    %% Initialize
    for d=1:D
       if T(d)==4 % discrete
            W(d,1:4)=[100 100 100 0]; %hyperparameters for the dirichlet prior on the datatypes weights
       elseif T(d)==3 % binary
            W(d,1:4)=[0 0 0 0];
       elseif T(d)==2 % continuous (only real and interval active)           
            W(d,:)=[100 100 0 0];
            X(X(:,d)==0,d)=1e-6; %to avoid numerical errors in positive real data
       elseif T(d)==1 %continuous with option positive
            W(d,:)=[100 100 0 100];
            X(X(:,d)==0,d)=1e-6; %to avoid numerical errors in positive real data
       end  
    end
    [Kest West countErr LIK]= DataTypes(Xmiss,T,R,W,1,1,1,0.001,1,Nits,KK, X);
    if countErr==0 % No numerica errors occur during inference
    output=[ouptutFold 'KK' num2str(KK) '_Nsim' num2str(Nits) '_it' num2str(it) '.mat'];
    save (output,'West', 'LIK');
    end
end