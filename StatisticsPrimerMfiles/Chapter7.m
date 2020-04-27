% CHAPTER 7

% PAGE 186

% Generate some multivariate Normal data
% Set up a vector of means.
mu = [2 -2];

% Specify the covariance matrix.
Sigma = [.9 .4; .4 .3];
X = mvnrnd(mu, Sigma, 200);
plot(X(:,1),X(:,2),'.');
xlabel('X_1')
ylabel('X_2')
title('Correlated Multivariate Normal Data')

% PAGE 187

% First get the covariance matrix.
% This will be the input to eig.
covm = cov(X);
% Get the number of rows and columns
size(covm)

% Demonstrate that this is symmetric. If it is,
% then the transpose is equal to itself.
isequal(covm,covm')

% It is nonsingular because the inverse exits.
covmI = inv(covm)

% What is the determinant?
det_cov = det(covm)

% PAGE 188

% Next get the eigenvalues and eigenvectors.
[A,D] = eig(covm);

% Display the elements of D.
display(D)

% Now project onto the new space and plot-Fig 7.1.
% Just data transformation at this point.
Z = X*A;
plot(Z(:,1),Z(:,2),'.')
xlabel('PC 1')
ylabel('PC 2')
title('Data Transformed to PC Axes')

% Get the covariance or the PC scores.
cov(Z)

% PAGE 190

% Get the sum of the original variances
sum(diag(covm))

% Get the sum of the eigenvalues
sum(diag(D))

% PAGE 191

% Load the iris data and
% put into one matrix.
load iris
X = [setosa; versicolor; virginica];

% Use the PCA function that inputs raw data.
% The outputs match previous eigenanalysis notation.
[A_pca,Z_pca,vl_pca] = pca(X);

% PAGE 192

% Use the PCA function that takes the
% covariance matrix as input
cov_iris = cov(X);
[A_pcacv,vl_pcacv,var_exp] = pcacov(cov_iris);

% Display the percent variance explained.
var_exp

% PAGE 193

% Create a scree plot.
plot(1:4,vl_pcacv,'-o')
axis([0.75 4 0 4.5])
xlabel('Index')
ylabel('lambda _j')
title('Scree Plot for Fisher''s Iris Data')

% PAGE 194

% The scree plot indicates that we should keep two
% dimensions. This explains the following
% percentage of the variance.
sum(var_exp(1:2))

% Get a biplot for the iris data.
varlab = {'S-Length','S-Width',...
    'P-Length','P-Width'};
Z_pcacv = X*A_pcacv;
figure,biplot(A_pcacv(:,1:2),...
    'scores',Z_pcacv(:,1:2),...
    'VarLabels',varlab)
ax = axis; axis([-.2 1 ax(3:4)])
box on

% PAGE 198

load ustemps

% Put the locations into a data matrix.
X = [-Long(:),Lat(:)];

% Plot the latitude and longitude as a scatter plot.
plot(-Long,Lat,'*')
axis([-130 -65 23 50])
xlabel('Longitude'), ylabel('Latitude')
title('Location of US Cities')

% Select some cities to display text labels
ind = [6,10,12,16,24,26,53];

% The City object is imported with the data.
text(X(ind,1),X(ind,2),City(ind))

% Find the distances between all points.
Dv = pdist(X,'euclidean');

% PAGE 200

% We want a 2-D embedding or configuration.
Y = cmdscale(Dv,2);

% Construct a scatter plot.
figure,plot(Y(:,1),Y(:,2),'.')
xlabel('C-MDS Dimension 1')
ylabel('C-MDS Dimension 2')
title('Classical MDS Configuration for US Cities')

% Need to rotate to match the orientation.
view([180,90])
text(Y(ind,1),Y(ind,2),City(ind))

% PAGE 202

% Load the data... just in case.
load iris
% Put the three variables into one data matrix.
X = [setosa; versicolor; virginica];

% PAGE 203

% We need to get the dissimilarities for inputs.
% Let's use the City Block distance as a change.
Dv = pdist(X,'cityblock');

% We are ready for metric MDS.
% We will use the 'metricsstress' criterion.
Xd = mdscale(Dv,2,'criterion','metricsstress');

% First create a vector of group IDs.
G = [ones(1,50),2*ones(1,50),3*ones(1,50)];

% Now get the grouped scatter plot.
% This is in the Statistics Toolbox.
gscatter(Xd(:,1),Xd(:,2),G,[],'.od')
box on
xlabel('Metric MDS Coordinate 1')
ylabel('Metric MDS Coordinate 2')
title('Embedding of Iris Data Using Metric MDS')

% Alternative embedding.
[Xd,stress]=mdscale(Dv,2,'criterion','metricsstress');

% PAGE 206

% The correlations of Crime Rates are from Borg
% and Groenen [2005]. They are saved in crime.mat.
load crime

% Now get a configuration of points in two
% dimensions using nonmetric MDS.
[Xd,stress] = mdscale(crime,2,'criterion','stress');

% Plot the points in 2-D.
plot(Xd(:,1),Xd(:,2),'o')
xlabel('MDS 1')
ylabel('MDS 2')
title('Nonmetric MDS of Crime Rates')
text(Xd(:,1)+0.025,Xd(:,2),crim_lab)
axis([-0.4 .8 -0.3 0.4])

% PAGE 207

% Load the data.
load countries
% Perform nonmetric MDS using the squared stress.
% We also get the disparities as output.
[Xd,stress,dispar] = ...
mdscale(nat,2,'criterion','sstress');
plot(Xd(:,1),Xd(:,2),'.')
text(Xd(:,1)+0.01,Xd(:,2)+.02,nat_lab)
title('Nonmetric MDS of Nation Similarities (1971)')
xlabel('MDS 1')
ylabel('MDS 2')

% PAGE 208-209

% Construct a Shepard diagram.
% First get the Euclidean distances in the
% MDS space.
dist = pdist(Xd);

% Get unique values below the diagonal of
% disparities matrix.
dispar = squareform(dispar);

% Need to convert input to dissimilarities.
natD = sqrt(1 - nat);

% Converts to vector form.
natV = squareform(natD);

% Now create the plot.
[dum,ord] = sortrows([dispar(:) natV(:)]);
plot(natV,dist,'x',...
natV(ord),dispar(ord),'*-')
xlabel('Input Dissimilarities')
ylabel('Distances and Disparities')
legend({'Distances','Disparities'})
title('Shepard Diagram')

% PAGE 211
load iris
% Put the three variables into one data matrix.
X = [setosa; versicolor; virginica];
% Create a scatter plot matrix
plotmatrix(X)

% Create a plot matrix of January temperature
% against Latitude and Longitude.
load ustemps
dat = [Lat,Long];
plotmatrix(dat,JanTemp)

% PAGE 212 - 213
load iris
% Put into one matrix.
X = [setosa; versicolor; virginica];
% Use the PCA function that takes raw data.
[A_pca,Z_pca,vl_pca] = pca(X);

% Create a grouping variable and a vector of names.
g = [ones(50,1);2*ones(50,1);3*ones(50,1)];
PCNames = {'PC_1','PC_2','PC_3','PC_4'}
gplotmatrix(Z_pca,[],g,[],'.ox',4,...
'off','variable',PCNames)
title('Fisher''s Iris Data in Principal Components')

% PAGE 215-216
% Show the Fisher's iris data in parallel coordinates.
% Load data and put into one matrix.
load iris
X = [setosa; versicolor; virginica];
parallelcoords(X)
box on
title(...
    'Parallel Coordinate Plot of Fisher''s Iris Data')

% PAGE 217 - 218
% Show Fisher's iris data as Andrews curves
andrewsplot(X)
box on
title('Andrews Plot of Fisher''s Iris Data')


