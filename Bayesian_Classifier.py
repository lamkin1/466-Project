import numpy as np
# Modified Lab 2 Code

def compute_priors_lab2(y):
    priors = {}
    total_samples = len(y)

    for label in y:
        priors[label] = priors.get(label, 0) + 1

    sorted_labels = sorted(priors.keys())
    priors = {f"{y.name}={label}": priors[label] / total_samples for label in sorted_labels}

    return priors

def specific_class_conditional_lab2(x, xv, y, yv):
    count = np.sum((x == xv) & (y == yv))
    total_samples = np.sum(y == yv)

    if count != 0:
        prob = count / total_samples
    else:
        prob = 0.01

    return prob

def class_conditional_lab2(X, y):
    probs = {}

    y_values = np.unique(y)

    for feature in X.columns:
        feature_values = np.unique(X[feature])
        for value in feature_values:
            condition = f"{feature}={value}"
            for y_value in y_values:
                condition_key = f"{condition}|{y.name}={y_value}"
                probs[condition_key] = specific_class_conditional_lab2(X[feature], value, y, y_value)

    return probs

def posteriors(probs, priors, x):
    likelihoods = {}

    cond_list = "|" + ",".join(f"{key}={(x[key])}" for key in x.keys())

    for class_label in priors.keys():
        likelihood = 1.0
        for key in x.keys():
            likelihood *= probs.get(f'{key}={(x[key])}|{class_label}', 0)
        likelihoods[class_label] = likelihood

    denominator = sum(likelihoods.values())

    # Check if the denominator is zero or very small (numerically unstable)
    if denominator <= np.finfo(float).eps:
        # Set posterior probabilities to 0.5 for all class labels if the combination doesn't exist or has very low probability
        post_probs = {f'{label}{cond_list}': 0.5 for label in priors.keys()}
    else:
        # Calculate posterior probabilities normally
        post_probs = {f'{label}{cond_list}': likelihoods[label] * priors[label] / denominator for label in priors.keys()}

    return post_probs

def train_test_split(X, y, test_frac=0.5):
    inxs = list(range(len(y)))
    np.random.shuffle(inxs)
    X = X.iloc[inxs, :]
    y = y.iloc[inxs]
    
    split_index = int(len(y) * test_frac)
    Xtrain, ytrain = X.iloc[:-split_index, :], y.iloc[:-split_index]
    Xtest, ytest = X.iloc[-split_index:, :], y.iloc[-split_index:]
    
    return Xtrain, ytrain, Xtest, ytest

def exercise_6_lab2(Xtrain, ytrain, Xtest, ytest, prior_type='uniform'):
    priors = compute_priors_lab2(ytrain)
    probs = class_conditional_lab2(Xtrain, ytrain)
    correct_predictions = 0

    for i in range(len(Xtest)):
        x = Xtest.iloc[i, :]
        post_probs = posteriors(probs, priors, x)

        predicted_class = max(post_probs, key=post_probs.get)

        if predicted_class.split("|")[0].strip().split("=")[1] == ytest.iloc[i]:  # Compare predicted_class as string
            correct_predictions += 1

    accuracy = correct_predictions / len(Xtest)
    return accuracy

def exercise_7_lab2(Xtrain, ytrain, Xtest, ytest, npermutations=10):
    # Initialize the dictionary to store importances
    importances = {}
    for col in Xtrain.columns:
        importances[col] = 0
    # Find the original accuracy
    orig_accuracy = exercise_6_lab2(Xtrain, ytrain, Xtest, ytest)
    # Carry out feature importance calculations
    for col in Xtrain.columns:
        for perm in range(npermutations):
            Xtest2 = Xtest.copy()
            Xtest2[col] = Xtest2[col].sample(frac=1, replace=False).values
            accuracy = exercise_6_lab2(Xtrain, ytrain, Xtest2, ytest)
            importances[col] += orig_accuracy - accuracy
    # Calculate the average importance
        importances[col] /= npermutations
    return importances

def exercise_8_lab2(Xtrain, ytrain, Xtest, ytest, npermutations=20):
    # initialize what we are going to return
    importances = {}
    for col in Xtrain.columns:
        importances[col] = 0

    # find the original accuracy
    orig_accuracy = exercise_6_lab2(Xtrain, ytrain, Xtest, ytest)

    # now carry out the feature importance work
    for col in Xtrain.columns:
        for perm in range(npermutations):
            Xtrain2 = Xtrain.copy()
            Xtrain2[col] = Xtrain[col].sample(frac=1, replace=False).values
            # Train and evaluate the Bayesian classifier with the modified feature
            accuracy = exercise_6_lab2(Xtrain2, ytrain, Xtest, ytest)
            importances[col] += orig_accuracy - accuracy
                
    for col in Xtrain.columns:
        importances[col] = importances[col] / npermutations

    return importances

# Gaussian Code

def compute_priors(y, prior_type='uniform'):
    priors = {}
    total_samples = len(y)

    if prior_type == 'uniform':
        prior_value = 1 / len(np.unique(y))
        for label in y:
            priors[f"{y.name}={label}"] = prior_value
    elif prior_type == 'empirical':
        for label in y:
            priors[label] = priors.get(label, 0) + 1
        priors = {f"{y.name}={label}": priors[label] / total_samples for label in priors.keys()}
    else:
        raise ValueError("Invalid prior_type. Use 'uniform' or 'empirical'.")

    return priors

def gaussian_likelihood(x, mean, std):
    exponent = -0.5 * ((x - mean) / std) ** 2
    return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(exponent)

def class_conditional_gaussian(X, y):
    class_cond_probs = {}

    for y_value in np.unique(y):
        X_given_y = X[y == y_value]
        class_cond_probs[f"{y.name}={y_value}"] = {
            feature: {
                'mean': np.mean(X_given_y[feature]),
                'std': np.std(X_given_y[feature]) + 0.0001
            }
            for feature in X.columns
        }

    return class_cond_probs

def posteriors_gaussian(class_cond_probs, priors, x):
    post_probs = {}
    for y_value in class_cond_probs.keys():
        posterior = priors[y_value]
        for feature in x.keys():
            likelihood = gaussian_likelihood(x[feature], class_cond_probs[y_value][feature]['mean'], class_cond_probs[y_value][feature]['std'])
            posterior *= likelihood
        post_probs[y_value] = posterior

    return post_probs

def exercise_6_gaussian(Xtrain, ytrain, Xtest, ytest, prior_type='uniform'):
    priors = compute_priors(ytrain, prior_type)
    class_cond_probs = class_conditional_gaussian(Xtrain, ytrain)
    correct_predictions = 0

    for i in range(len(Xtest)):
        x = Xtest.iloc[i, :]
        post_probs = posteriors_gaussian(class_cond_probs, priors, x)

        predicted_class = max(post_probs, key=post_probs.get)
        
        if predicted_class.split("=")[1] == ytest.iloc[i]:
            correct_predictions += 1

    accuracy = correct_predictions / len(Xtest)
    return accuracy

def exercise_7_gaussian(Xtrain, ytrain, Xtest, ytest, npermutations=10):
    # Initialize the dictionary to store importances
    importances = {}
    for col in Xtrain.columns:
        importances[col] = 0
    # Find the original accuracy
    orig_accuracy = exercise_6_gaussian(Xtrain, ytrain, Xtest, ytest)
    # Carry out feature importance calculations
    for col in Xtrain.columns:
        for perm in range(npermutations):
            Xtest2 = Xtest.copy()
            Xtest2[col] = Xtest2[col].sample(frac=1, replace=False).values
            accuracy = exercise_6_gaussian(Xtrain, ytrain, Xtest2, ytest)
            importances[col] += orig_accuracy - accuracy
    # Calculate the average importance
        importances[col] /= npermutations
    return importances

def exercise_8_gaussian(Xtrain, ytrain, Xtest, ytest, npermutations=20):
    # initialize what we are going to return
    importances = {}
    for col in Xtrain.columns:
        importances[col] = 0

    # find the original accuracy
    orig_accuracy = exercise_6_gaussian(Xtrain, ytrain, Xtest, ytest)

    # now carry out the feature importance work
    for col in Xtrain.columns:
        for perm in range(npermutations):
            Xtrain2 = Xtrain.copy()
            Xtrain2[col] = Xtrain[col].sample(frac=1, replace=False).values
            # Train and evaluate the Bayesian classifier with the modified feature
            accuracy = exercise_6_gaussian(Xtrain2, ytrain, Xtest, ytest)
            importances[col] += orig_accuracy - accuracy
                
    for col in Xtrain.columns:
        importances[col] = importances[col] / npermutations

    return importances