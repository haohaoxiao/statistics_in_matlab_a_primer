% Chapter 3

% PAGE 55
% We will first load the variable.
load earth

% Now, we find the mean.
xbar = mean(earth)

% Find the median density of the Earth.
med = median(earth)

% PAGE 56
% Find the trimmed mean density of the Earth.
% We will use 20% for our trimming.
xbar_trim = trimmean(earth, 20)

% Now, find the mode.
mod_earth = mode(earth)

% PAGEs 59 - 60

% First find the minimum, maximum and range
% of the earth data.
minearth = min(earth)
maxearth = max(earth)

% Find the range. 
rngearth = range(earth)

% We can also find the range, as follows
maxearth - minearth

% Next, we find the variance and the
% standard deviation.
vearth = var(earth)
searth = std(earth)

% We can also find the standard deviation
% by taking the square root of the variance.
sqrt(var(earth))

% PAGE 61

% Now, let's find the covariance matrix of the
% setosa iris. First, we need to load it.
load iris

% Find the covariance matrix of the setosa object.
cvsetosa = cov(setosa)

% Find the correlation coefficients.
crsetosa = corrcoef(setosa)

% If the argument to the var function is a matrix,
% then it will return the variance of each column
% or variable.
var(setosa)

% PAGE 65

% First, we find the quartiles.
quart = quantile(earth,[.25 .50 .75])

% This is what we get from the median function.
median(earth) % This gives the same result.

% Next, find the IQR of the earth data.
iqrearth = iqr(earth)

% PAGE 66

% Get the quartiles using a different function. 
pearth = prctile(earth,[25 50 75])

% Find the sample skewness of the earth data.
skearth = skewness(earth)

% PAGE 69

% Construct histogram using 10 bins.
% This is in base MATLAB.
hist(earth)
title('Histogram of the Earth Density Data')
xlabel('Multiple of the Density of Water')
ylabel('Frequency')

% PAGE 71

% Use the histfit function in the
% Statistics Toolbox.
% Request a kernel density estimate along with
% the histogram, and use more bins.
histfit(earth,20,'kernel')
title('Histogram and Density of the Earth Data')
xlabel('Multiple of the Density of Water')
ylabel('Frequency')

% PAGE 72

% Get a probability plot comparing the
% earth data to a normal distribution.
probplot(earth)
xlabel('Earth Quantiles')

% Construct a Q-Q plot of the sepal length
% for Virginica and Versicolor in the iris data.
qqplot(virginica(:,1),versicolor(:,1))
title('Q-Q Plot of Sepal Length - Iris Data')
xlabel('Virginica Quantiles')
ylabel('Versicolor Quantiles')

% Create boxplots of sepal length
% for the iris data.
boxplot([setosa(:,1),virginica(:,1),...
    versicolor(:,1)],...
    'notch','on',...
    'labels',{'Setosa','Virginica','Versicolor'})
title('Notched Boxplots of Sepal Length')



















