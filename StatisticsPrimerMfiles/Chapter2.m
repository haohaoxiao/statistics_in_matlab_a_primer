% Chapter 2

% PAGE 40 - 41

% To illustrate how to plot a curve, first
% generate the values for a normal distribution.
% This creates a standard normal probability
% distribution object.
stdnpd = makedist('Normal');

% Now, create one with different parameters.
npd = makedist('Normal','mu',0,'sigma',2);

% Define a vector for the domain.
x = -4:.01:4;

% Get the y values for both distributions.
y1 = pdf(stdnpd, x);
y2 = pdf(npd, x);

% Now, plot the curves on the same set of axes.
plot(x,y1,x,y2,'-.')
xlabel('X')
ylabel('PDF')
title('Plots of Normal Distributions')
legend('mu = 0, sigma = 1','mu = 0, sigma = 2')

% PAGE 42

% Get the vector for the mean.
mu = [0 0];

% Get the covariance matrix.
sigma = eye(2);

% Obtain the (x,y) pairs for the domain.
x = -3:.2:3; y = -3:.2:3;
[X,Y] = meshgrid(x,y);

% Evaluate the multivariate normal at
% the coordinates.
Z = mvnpdf([X(:), Y(:)],mu,sigma);

% Reshape to a matrix.
Z = reshape(Z,length(x),length(y));

% The surface plot is shown in Figure 2.3.
% Now, create the surface plot and add labels.
surf(X,Y,Z);
xlabel('X'), ylabel('Y'), zlabel('PDF')
title('Multivariate Normal Distribution')
axis tight

% PAGE 44

% Load the UStemps and iris data.
load UStemps
load iris

% Construct a 2-D scatter plot with plot,
% using temperature and latitude.
plot(Lat, JanTemp, '*')

% Adjust the axes to change white space.
axis([24 50 -2 70])

% Add labels.
xlabel('Latitude')
ylabel('Temperature (degs)')
title('Average January Temperature - US Cities')

% PAGE 45

% The next example shows how to construct a 3-D scatter plot using plot3.
% The scatter plot is shown in Figure 2.5.
% Construct a 3-D scatter plot using plot3.
plot3(Long, Lat, JanTemp, 'o')

% Add a box and grid lines to the plot.
box on
grid on

% Add labels.
xlabel('Longitude')
ylabel('Latitude')
zlabel('Temperature (degs)')
title('Average January Temperature - US Cities')

% PAGE 45

% This is an example of a scatter plot matrix of the iris data. 
% See Figure 2.6 for the results.
% This produces a scatter plot matrix.
% We first need to put the data into one matrix.
irisAll = [setosa; versicolor; virginica];
plotmatrix(irisAll)
title('Iris Data - Setosa, Versicolor, Virginica')







