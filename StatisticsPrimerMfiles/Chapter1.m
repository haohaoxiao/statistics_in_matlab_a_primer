% Chapter 1

% PAGE 9

% First see what is in the workspace. This command
% lists all variables in the workspace.
who

% The workspace is empty and nothing is returned.
% Now load the iris.mat file.
load iris

% What is in the workspace now?
who

% PAGE 10

% Now save the variables in another file.
% We will save just the setosa variable object.
% Use save filename varname.
save setosa setosa

% See what files are in the current directory.
dir

% Remove objects from workspace to clean it up.
clear
% The directory should be empty. Load the earth data.
load DensityEarth.txt -ascii
% See what is in the workspace.
who

% PAGE 16

% Create a vector x.
x = [2, 4, 6];
% Delete the second element.
x(2) = [];
% Display the vector x.
disp(x)

% Find the elements that are negative.
ind = find(x < 0);

% Print the vector ind to the command window.
ind

% PAGE 17

% Create a cell array, where one cell contains
% numbers and another cell element is a string.
cell_arry2 = {[1,2], 'This is a string'};
% Let’s check the size of the cell array
size(cell_arry2)

% PAGE 18
% Create a structure called employee with three fields.
employee = struct(...
    'name',{{'Wendy','MoonJung'}},...
    'area',{{'Visualization','Inference'}},...
    'deg',{{'PhD','PhD'}},...
    'score',[90 100])

all_names = employee.name


% PAGE 19

load UStemps
% Create a table using all four variables.
UTs_tab = table(City,JanTemp,Lat,Long)

% See what is in the workspace.
whos


% PAGE 20

% Extract the employee's area of interest.
e_area = employee.area;

% Display the contents in the window.
e_area

% Display Wendy's score.
employee.score(1)

% Get MoonJung's area.
employee.area{2}

% PAGE 21

% Get a partial table by extracting the first
% three rows.
U1 = UTs_tab(1:3,:)

% Get the JanTemp variable.
jt = UTs_tab.JanTemp;

% See if it is equal to the JanTemp variable.
isequal(jt,JanTemp)

% We get an answer of 1, indicating they are the same.

% We can extract the Lat and Long data for the first
% three cities using the variable names.
U2 = UTs_tab{1:3,{'Lat','Long'}}

% PAGE 23

% First load the data back into the workspace.
load iris

% Now, we use the semicolon to stack
% setosa, versicolor, and virginica data objects.
irisAll = [setosa; versicolor; virginica];

% Now look at the workspace to see what is there now.
who

% Check on the size of irisAll.
size(irisAll)

% PAGE 24

% Load the data if not in the workspace.
load UStemps

% Use commas to concatenate as a row.
UStemps = [JanTemp, Lat, Long];

% Check the workspace.
who

% Verify the size of UStemps:
size(UStemps)


% PAGE 32

% Save the data for the UStemps to a .mat file.
save UStemps City JanTemp Lat Long

% That used the command syntax to call a function.
% Now use the function syntax.
save('USt.mat','City','JanTemp','Lat','Long')

% See what is in our directory now.
dir



