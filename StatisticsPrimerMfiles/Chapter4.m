% Chapter 4

% PAGE 85
% Generate a set of normal random variables.
% The mean is 0 and sigma is 1.
n = 10;
x = normrnd(0,1,1,n);

% Get a grid of points to evaluate density.
pts = linspace(-4,4);

% Set up a vector for our estimated PDF.
kpdf = zeros(size(pts));

% We need a bandwidth for the kernels.
% This is the normal reference rule [Scott, 1992].
h = 1.06*n^(-1/5);
% Hold the plot, because we will add
% a curve at each iteration of the loop.
hold on
for i = 1:n
    
    % Use the normal kernel, noting that
    % the mean is given by an observation
    % and the standard deviation is h.
    f = normpdf(pts, x(i), h);
    
    % Plot the kernel function for i-th point.
    plot(pts, f/n);
    
    % Keep adding the individual kernel function.
    kpdf = kpdf + f/n;
end

% Plot the kernel density estimate.
plot(pts, kpdf)
title('Kernel Density Estimate')
xlabel('X')
ylabel('PDF')
hold off

% PAGE 87

% First, generate some points for the domain.
pts = linspace(-4,4);

% Obtain a standard normal PDF.
nrmpdf = normpdf(pts,0,1);

% Obtain a Student's t with v = 1.
stpdf = tpdf(pts, 1);

% Plot the two PDFs.
plot(pts,nrmpdf,'-',pts,stpdf,'-.')
axis([-4 4, 0 0.42])
title('Normal and Student''s t Distributions')
xlabel('x')
ylabel('PDF')
legend('Normal','Student''s t')

% PAGE 88

% First specify the parameter for the Poisson.
lambda = 3;

% Get the points in the domain for plotting
pts = 0:10;

% Get the values of the PDF
ppdf = poisspdf(pts,lambda);

% Get the values of the CDF
pcdf = poisscdf(pts,lambda);

% Construct the plots
subplot(2,1,1)
plot(pts, ppdf, '+')
ylabel('Poisson PDF')
title('Poisson Distribution \lambda = 3')
subplot(2,1,2)
stairs(pts, pcdf)
ylabel('Poisson CDF')
xlabel('X')

% PAGE 90

% Example - Multivariate (2-D) Student's t
% First, set the parameters.
corrm = [1, 0.3 ; 0.3, 1];
df = 2;

% Get a grid for evaluation and plotting
x = linspace(-2,2,25);
[x1,x2] = meshgrid(x,x);
X = [x1(:), x2(:)];

% Evaluate the multivariate Student's t PDF
% at points given by X.
pmvt = mvtpdf(X, corrm, df);

% Plot as a mesh surface.
mesh(x1,x2,reshape(pmvt,size(x1)));
xlabel('X1'), ylabel('X2'), zlabel('PDF')
title('2-D Student''s t Distribution')

% PAGE 91

% First load the data into the workspace.
load earth
n = length(earth);

% Set two bandwidths.
% Normal reference rule - standard deviation.
h1 = 1.06*std(earth)*n^(-1/5);

% Normal reference rule - IQR.
h2 = 0.786*iqr(earth)*n^(-1/5);

% Get a domain for the for the PDF.
[kd1,pts] = ksdensity(earth,'bandwidth',h1);
[kd2,pts] = ksdensity(earth,'bandwidth',h2);
plot(pts,kd1,pts,kd2,'-.')
legend('h1 = 0.1832','h2 = 0.1263')
title('Kernel Density Estimates - 2 Bandwidths')
ylabel('PDF'), xlabel('Density of Earth')

% PAGE 95

% Load the earth data
load earth
% Estimate the parameters of a normal distribution.
% Also ask for a 90% confidence interval
[mu1,sig1,muci1,sigci1] = normfit(earth,0.9);

% The following is the estimated mean.
display(mu1)

% Here is the estimated confidence interval.
display(muci1)

% The following estimated standard deviation is shown.
display(sig1)

% This is the confidence interval.
display(sigci1)

% Alternative way to estimate mean and standard deviation.
% Get the mean from the data.
mean(earth)

% Get the standard deviation
std(earth)

% PAGE 97

% Fit a normal distribution to the data.
pdfit = fitdist(earth,'normal')

% We can extract the confidence intervals using the paramci function. 
% This is one of the methods we can use with a ProbDist object.
% Display the confidence intervals only.
% The intervals are given in the columns.
paramci(pdfit)

% Fit a kernel density to the earth data.
ksfit = fitdist(earth,'kernel')

% PAGE 98

% Get a set of values for the domain.
pts = linspace(min(earth)-1,max(earth)+1);

% Get the PDF of the different fits.
pdfn = pdf(pdfit,pts);
pdfk = pdf(ksfit,pts);

% Plot both curves
plot(pts,pdfn,pts,pdfk,'--')

% Plot the points on the horizontal axis.
hold on
plot(earth,zeros(1,n),'+')
hold off
legend('Normal Fit','Kernel Fit')
title('Estimated PDFs for the Earth Data')
xlabel('Density of the Earth')
ylabel('PDF')

% PAGEs 99 - 100

% Example of multivariate normal using the iris data.
load iris

% Construct a scatter plot matrix.
% This function is in base MATLAB.
plotmatrix(virginica)
title('Iris Virginica')


% Extract the variables from the Virginica species.
X = virginica(:,[1,3]);
% Estimate the mean.
mu = mean(X);

% Estimate the covariance.
cv = cov(X);

% Establish a grid for the PDF
x1 = linspace(min(X(:,1))-1,max(X(:,1))+1,25);
x2 = linspace(min(X(:,2))-1,max(X(:,2))+1,25);
[X1, X2] = meshgrid(x1,x2);

% Use the parameter estimates and generate the PDF.
pts = [X1(:),X2(:)];
Xpdf = mvnpdf(pts,mu,cv);

% Construct a mesh plot.
mesh(X1,X2,reshape(Xpdf,25,25))
title('Estimated PDF for Iris Virginica')
xlabel('Sepal Length'), ylabel('Petal Length')
zlabel('PDF')
axis tight

% PAGE 107

% Set the seed to 10.
rng(10)

% Generate 3 random variables.
r1 = rand(1,3)

% Generate 3 more random variables.
r2 = rand(1,3)

% Now, set the seed back to 10.
rng(10)

% Generate 6 random variables.
r3 = rand(1,6)

% PAGE 108

% Create a Q-Q plot of the earth data, where
% we compare it to an exponential distribution.
load earth

% Generate exponential variables.
% Use the mean of the data for the parameter.
rexp = exprnd(mean(earth),size(earth));

% Create the Q-Q plot.
plot(sort(earth),sort(rexp),'o')
xlabel('Data'),ylabel('Exponential')
title('Q-Q Plot of Earth Data and Exponential')

% Add a line that is estimated using quartiles.
% Get the first and third quartiles of the earth.
qeth = quantile(earth,[.25,.75]);

% Now, get the same for the exponential variables.
qexp = quantile(rexp,[.25,.75]);

% Fit a straight line. See Chapter 6 for more details.
p = polyfit(qeth,qexp,1);
pp = polyval(p,[5,max(earth)]);
hold on
plot([min(earth),max(earth)],pp)

% PAGE 110

% Generate two vectors of normal random variables
% with different means and variances.
% First is a variance of 4 and mean of 2.
x1 = randn(500,1)*sqrt(4) + 2;

% Next is a variance of 0.7 and a mean of -2.
x2 = randn(500,1)*sqrt(0.7) - 2;

% Construct a scatter plot.
plot(x1, x2, 'o')
title('Uncorrelated Normal Random Variables')
xlabel('X1'), ylabel('X2')

% Get the correlations
corrcoef([x1,x2])

% Generate bivariate correlated random variables.
mu = [2 -2];
covm = [1 1.25; 1.25 3];
X = mvnrnd(mu,covm,200);

% Show in a scatter plot.
scatter(X(:,1),X(:,2))
xlabel('X_1'),ylabel('X_2')
title('Correlated Bivariate Random Variables')





























