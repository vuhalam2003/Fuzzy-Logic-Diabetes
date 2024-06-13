% Load the data
data = readtable('D:\Downloads\archive\clean_data.csv');

% Separate features and labels
features = data(:, 1:end-1);
labels = data(:, end);

% Load the FIS
fis = readfis('diabetes.fis');

% Initialize variables
n = size(features, 1);
predicted_outcomes = zeros(n, 1);

% Generate inputs and predict outcomes
for i = 1:n
    % Extract input features for the current row
    input_data = table2array(features(i, :));
    
    % Apply the FIS to predict the outcome
    predicted_outcome = evalfis(fis, input_data);
    
    % Convert the predicted outcome to binary (0 or 1)
    if predicted_outcome > 0.5
        predicted_outcomes(i) = 1;
    else
        predicted_outcomes(i) = 0;
    end
end

% Calculate accuracy
accuracy = sum(predicted_outcomes == table2array(labels)) / n;
fprintf('Accuracy: %.2f%%\n', accuracy * 100);