import numpy as np;

y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])
yhat = np.array([1, 1, 1, -1, -1, -1, -1, -1, -1, -1])
correct = (y == yhat)
print(correct)
# it returns an array of true and false based if the prediction was correct or incorrect

# step 1: set the weight vector w, to uniform weights, where sum of all weights is 1 
# step 2: train a weighted week learner
# step 3: predict class labels 
weights = np.full(10, 0.1)
print(weights)
epsilon = np.mean(~correct) # this is calculating the classification error since that symbol switches the values 
# step 4: Compute the weighted error rate
print(epsilon)
# basically it gets the false values and comparing as a proportion to the total array
# so this is efffectively the error

# step 5: Compute the coefficient
alpha_j = 0.5 * np.log((1-epsilon) / epsilon)
print(alpha_j)


# step 6: Update the weights using the following equation w = w * exp(-aj * y * y)
update_if_correct = 0.1 * np.exp(-alpha_j * 1 * 1)
print(update_if_correct)

# Similarly, we will increase the ith weight if y predicted the label incorrectly, like this:
update_if_wrong_1 = 0.1 * np.exp(-alpha_j * 1 * -1)
print(update_if_wrong_1)

# Alternatively, it's like this:
update_if_wrong_2 = 0.1 * np.exp(-alpha_j * -1 * 1)
print(update_if_wrong_2)

# Now we use these values to update the weights as follows:
weights = np.where(correct == 1, update_if_correct, update_if_wrong_1)
print(weights)

# step 7: Normalize the weights w= w/(sum of all weights)
normalized_weights = weights / np.sum(weights)
print(normalized_weights)
