function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    h = X*theta; %The hypothesis is a vector, formed by multiplying the X matrix and the theta vector.
    
    errors = h - y; %difference between the 'h' vector and the 'y' vector
    
    %% The gradient computation would then be X' * error, 
    %% because the size of X' is (n x m), and the error vector is (m x 1). 
    %% The sum of the products is automatically computed, 
    %% and this gives you a (n x 1) result. 
    %% That's exactly what you want (it is the same size as theta).
    %% Font: Tom MosherMentor · 3 years ago · Edited
    
    theta_change = (alpha / m) * (X' * errors);
    theta = theta - theta_change;
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end
end
