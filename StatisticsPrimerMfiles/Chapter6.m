% Chapter 6

% PAGE 149

% Generate predictor and response data using
% the true relationship and adding noise.
x = -4.8:0.3:4.8;
y = -2*x.^2 + 2*x + 3 + 5*randn(size(x));

% Estimate the parameters.
p = polyfit(x,y,2)

% PAGE 151

% Evaluate the polynomial and plot.
yhat = polyval(p,x);
plot(x,y,'o',x,yhat)
title('Scatter Plot and Model')
xlabel('X')
ylabel('Y')

% PAGE 151 - Next Example

load UStemps
% Create a scatter plot.
plot(Lat,JanTemp,'o')

% Page 152

% Call the polyfit function with different outputs.
[ptemp,Stemp] = polyfit(Lat,JanTemp,1);

% Display the object with the estimated model.
display(Stemp)

% Generate some different X values.
xp = linspace(min(Lat),max(Lat),10);
% Get estimates using the model.
[ust_hat,delta] = polyval(ptemp,xp,Stemp);
plot(Lat,JanTemp,'o')
axis([24 50 -1 70])
xlabel('Latitude')
ylabel('Temperature \circ Fahrenheit')
title('Average Minimum Temperature vs Latitude')
hold on

% PAGE 153

% This plots the data and adds error bars.
errorbar(xp, ust_hat, delta), hold off

% PAGE 154

% This example uses the data generated from a
% second-order polynomial. Create a design matrix
% 'A' using a quadratic model. 'x' is the solution
% vector, so we specify the observed predictors.
pred = x(:);
A = [pred.^2, pred, ones(size(pred))];

% The response variable needs to be a column vector.
% We continue to use the alternative notation.
b = y(:);

% Find the solution using the left-divide operator.
x = A\b

% PAGE 155

% Create a scatter plot of temperature and longitude.
plot(Long,JanTemp,'.')
xlabel('Longitude')
title('Minimum January Temperature')
ylabel('Temperature \circ Fahrenheit')

% PAGE 157

% First step is to create the design matrix A.
% We will include a constant term and two
% first-order terms for latitude and longitude.
At = [ones(56,1), Lat, Long];
[x,stdx,mse] = lscov(At,JanTemp)

% PAGE 159

% We need our design matrix X.
% The regress function requires a column of ones.
X = [ones(56,1), Lat, Long];

% Call the regress function and ask for all of the output arguments.
[b,bint,r,rint,stats] = regress(JanTemp,X)

% Look at the statistics for the object.
display(stats)

% Here is the F statistic.
stats(3)

% PAGE 162

% We just need the predictor matrix.
Xp = [Lat,Long];
stats = regstats(JanTemp,Xp)

% Extract the estimated coefficients.
bhat = stats.beta

% Display results of F test.
stats.fstat

% Display the results of the t-test.
stats.tstat

% PAGE 166

% Fit the same model as before.
mdl = LinearModel.fit(Xp,JanTemp,'linear');

% Display the model.
disp(mdl)

% Create a dataset array from the variables.
% Put the response as the last column.
ds = dataset(Lat,Long,JanTemp);
lm = fitlm(ds,'linear');
disp(lm)

% PAGE 167

% Display the ANOVA table
tbl = anova(lm)

% Get the 95% confidence intervals for the 
% estimated coefficients.
bint = coefCI(lm)

% PAGE 169

% Fit a straight line between temperature and latitude.
% We are using our dataset object from before.
lm1 = fitlm(ds,'JanTemp ~ Lat');

% Plot the model with the data.
plot(lm1)

% Get the R^2 from the linear model object.
lm1.Rsquared

% PAGE 170

% Test the coefficients.
coefTest(lm1)

% PAGE 171

% This time we use the predictor matrix and vector
% of responses as inputs to fitlm.
% Generate some normal RVs as a predictor variable.
xn = randn(size(JanTemp));

% Fit a first-order model.
lm2 = fitlm(xn, JanTemp, 'linear');
disp(lm2)

% Add the predictor that is just random
% noise and fit a multiple regression with first-order
% terms for Latitude and the noise predictor.
% Include latitude and noise in the predictor matrix.
Xn = [Lat, xn];
lm3 = fitlm(Xn,JanTemp,'linear');
disp(lm3)

% Get the confidence intervals for the coefficients.
coefCI(lm3)

% Look at the R^2.
lm3.Rsquared

% PAGE 173

% Adjust the model by adding terms for longitude.
% Repeating the linear model first...
mdl1 = fitlm(ds,'JanTemp ~ Lat + Long');

% Next, add a quadratic term in longitude.
mdl2 = fitlm(ds,'JanTemp ~ Lat + Long^2');

% Now, add a cubic term in longitude.
mdl3 = fitlm(ds,'JanTemp ~ Lat + Long^3');

% Get the R-squared values.
R1 = mdl1.Rsquared
R2 = mdl2.Rsquared
R3 = mdl3.Rsquared

% PAGE 174

% Construct a probability plot of the residuals.
plotResiduals(mdl3,'probability')

% Plot of the residuals against the fitted values.
plotResiduals(mdl3,'fitted')





















