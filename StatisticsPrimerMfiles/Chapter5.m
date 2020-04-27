% CHAPTER 5

% PAGE 120

mu0 = 1800;
sig = 200;
alpha = 0.05;
n = 50;
cv = norminv(alpha, mu0, sig/sqrt(n));

% We could also calculate the p-value using the CDF function:
pv = normcdf(1750, mu0, sig/sqrt(n));

% PAGE 121

% Let’s assume that we do not know the population
% standard deviation. In this case, we have to use
% the Student's t distribution for the test statistic.
% First, find the degrees of freedom.
df = n - 1;

% Find the critical value.
cv_t = tinv(alpha, df)


% We estimated the standard deviation from the data
% and it had a value of 175.
sig_hat = 175;
t_obs = (1750 - 1800)/(sig_hat/sqrt(n))

% PAGE 122

% Now, find the observed p-value.
pv = tcdf(t_obs,df)

% PAGE 123
% First do a boxplot of the sepal length.
load iris

% This is the first variable - extract it.
setSL = setosa(:,1);
virSL = virginica(:,1);
verSL = versicolor(:,1);

% Do a notched boxplot.
boxplot([setSL,virSL,verSL],'notch','on',...
'labels',{'Setosa','Virginica','Versicolor'})
ylabel('Sepal Length')
title('Boxplots of Sepal Length in Iris Data')

% PAGE 124

% As another check, do a normal probability plot.
normplot([setSL, virSL, verSL])

% Now, do a Lilliefors' test for normality.
lillietest(setSL)
lillietest(virSL)
lillietest(verSL)

% perform the hypothesis test.
[hyp, pv, ci] = ztest(setSL, 5, 0.3)


% PAGE 125

% An additional example with Versicolor
% sepal length and the alternative hypothesis that
% the mean is greater than 5.7.
[hyp,pv,ci] = ztest(verSL,5.7,0.6,'tail','right')


% PAGE 126

% Load fish prices data.
load fish

% Construct a normal probability plot to test
% the normality assumption.
normplot([fish70, fish80])

% Find the mean of the fish prices in 1970.
mu70 = mean(fish70);

% PAGE 127

% Do a t-test to see if the fish prices
% in 1980 have the same mean as in 1970.
[hyp, pv, ci] = ttest(fish80,mu70)

% PAGE 128
% Ask for the paired t-test.
[hyp, pv, ci] = ttest(fish70, fish80)

% Ask for the t-test.
hyp = ttest(fish80,mu70,'tail','right')

% PAGE 129
% Hypothesis test with the VarType argument in the function call.
[h,pv,ci] = ttest2(setSL, verSL,'VarType','unequal')


% PAGE 130

load UStemps % Saved after importing - Chapter 1
% This imports four variables or data objects into the workspace.
% Two are Lat and JanTemp. 
% Construct a scatter plot.
plot(Lat,JanTemp,'o')
xlabel('Latitude')
ylabel('Temperature ( \circ Fahrenheit )')
title('Average January Temperature - US Cities')

% Find correlation.
cr = corr(Lat, JanTemp)

% PAGE 132

% Get the bootstrap confidence interval.
ci = bootci(5000, @corr, Lat, JanTemp)

% Another version of the confidence interval.
ci = bootci(5000, {@corr, Lat, JanTemp},...
    'type','normal')

% PAGE 133

% Get the bootstrap replicates.
bootrep = bootstrp(5000,@corr,Lat,JanTemp);

% Show their distribution in a histogram.
hist(bootrep,25)
xlabel('Estimated Correlation Coefficient')
ylabel('Frequency')
title('Histogram of Bootstrap Replicates')

% We could also obtain an estimate of the standard error for the correlation.
se = std(bootrep)

% PAGE 137
load iris

% This gives us three separate data objects in
% the workspace — one for each species.
% We need to put one of the characteristics
% from all three groups into one matrix.
% Let's look at sepal width ...
sepW = [setosa(:,2),virginica(:,2),versicolor(:,2)];

% Perform a one-way ANOVA.
[pval, anova_tbl, stats] = anova1(sepW)

% PAGE 139
% Perform a multiple comparison.
comp = multcompare(stats)















