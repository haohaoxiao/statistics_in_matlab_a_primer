% CHAPTER 8

% PAGE 225
% Load the iris data. This imports three data objects:
% setosa, versicolor, and virginica.
load iris
% Create a data matrix.
X = [setosa; versicolor; virginica];
% Create a vector of class labels.
y = [ones(50,1); 2*ones(50,1); 3*ones(50,1)];

% Create feature names.
ftns = {'S-Length','S-Width','P-Length','P-Width'};
% Create a linear classifier.
% The 'DiscrimType' default is 'linear'.
% The 'Prior' default is 'empirical'.
% We specify them here for clarity.
Lclass = fitcdiscr(X,y,'DiscrimType','linear',...
    'PredictorNames',ftns,...
    'Prior','empirical');
% Create a quadratic classifier.
Qclass = fitcdiscr(X,y,'DiscrimType','quadratic',...
    'PredictorNames',ftns,...
    'Prior','empirical');

% PAGE 226
% Calculate and show the error for the linear
% classifier.
Lrerr = resubLoss(Lclass)

% Calculate and display the error for the quadratic
% classifier.
Qrerr = resubLoss(Qclass)

% PAGE 227
% Get the cross-validation (CV) prediction error.
% The crossval function creates a CV object.
Lcv_mod = crossval(Lclass,'kfold',5);
Qcv_mod = crossval(Qclass,'kfold',5);

% Now get the error based on CV.
Lcverr = kfoldLoss(Lcv_mod)
Qcverr = kfoldLoss(Qcv_mod)

% resubPredict provides the resubstitution response
% of a classifier.
R = confusionmat(Qclass.Y,resubPredict(Qclass))

% PAGE 228
% We can get the number of misclassified points for
% the linear classifier by multiplying the error by n.
Lrerr*150

% PAGE 229
% Use iris data and kernel option.
NBk = fitNaiveBayes(X,y,'distribution','kernel')

% We can also check class of the object.
class(NBk)

% Get the confusion matrix for this case.
predNBk = predict(NBk,X);
confusionmat(y,predNBk)

% PAGE 230
% Also get the Gaussian and compare the error.
% The 'normal' is the default, but we specify it
% for clarity.
NBn = fitNaiveBayes(X,y,'distribution','normal')
% Get the confusion matrix for this case.
predNBn = predict(NBn,X);
confusionmat(y,predNBn)

% PAGE 232
% Use the same iris data from before.
Knn = fitcknn(X,y,'NumNeighbors',5)

% Get the resubstitution error.
Krerr = resubLoss(Knn)

% Get the cross-validation object first.
Kcv_mod = crossval(Knn,'kfold',5);

% Now get the error based on cross-validation.
Kcverr = kfoldLoss(Kcv_mod)

% PAGE 233
% Get the confusion matrix.
R = confusionmat(Knn.Y,resubPredict(Knn))

% PAGE 238
% Create some two-cluster data.
% Use this to create a dendrogram.
X = [randn(20,2)+2; randn(10,2)-2];
plot(X(:,1),X(:,2),'.')
% Get the dendrogram using the raw data,
% complete linkage, and Euclidean distance.
Z = linkage(X,'complete');
dendrogram(Z)
title('Dendrogram Example')

% PAGE 239
% Load the data.
load iris
% Create a data matrix.
X = [setosa; versicolor; virginica];
% Agglomerative clustering with defaults
% of Euclidean distance and single linkage.
% First, find the Euclidean distance.
Euc_d = pdist(X);
% Get single linkage, which is the default.
Zs = linkage(Euc_d);
% Construct the dendrogram.
dendrogram(Zs)
title('Single Linkage - Iris Data')

% Try complete linkage.
Zc = linkage(Euc_d,'complete');
dendrogram(Zc)
title('Complete Linkage - Iris Data')

% Get three clusters from each method.
cidSL = cluster(Zs,'maxclust',3);
cidCL = cluster(Zc,'maxclust',3);

% The tabulate function provides a summary of the % cluster IDs.
tabulate(cidSL)
tabulate(cidCL)

% PAGE 241
% Use the complete linkage clusters.
varNames={'S-Length','S-Width','P-Length','P-Width'}
gplotmatrix(X,[],cidCL,[],...
    '.ox',3.5,'off','variable',varNames);
title('Clusters in Iris Data - Complete Linkage')

% PAGE 244
% Load the data
load iris
% Create a data matrix.
X = [setosa; versicolor; virginica];

% Apply K-means clustering using the
% 'cosine' distance and a random start
% selected uniformly from the range of X.
cidK3 = kmeans(X,3,...
    'distance','cosine',...
    'start','uniform');

% Look at the results.
tabulate(cidK3)

% PAGE 245
% Get a silhouette plot for K-means output.
silhouette(X,cidK3,'cosine');
title('Silhouette Plot - K-means')

% Get a silhouette plot for the agglomerative
% clusters. That clustering used the default
% squared Euclidean distance.
silhouette(X,cidCL);
title('Silhouette Plot - Complete Linkage')

% PAGE 247
% Get the average silhouette value of K-means clusters.
sK = silhouette(X,cidK3,'cosine');
msK = mean(sK)

% Get the average silhouette value of agglomerative
% clustering with complete linkage.
sC = silhouette(X,cidCL);
msC = mean(sC)

% Get replicates to find the 'best' solution.
% Use the default starting point, 6 replicates,
% and display the results of each iteration.
cidK4 = kmeans(X,4,...
    'distance','cityblock',...
    'display','final',...
    'replicates',6);

% Get the average silhouette value.
sK4 = silhouette(X,cidK4,'cosine');
msK4 = mean(sK4)


