import numpy as np

# Assume cm is your confusion matrix
cm = np.array([[11, 4, 0, 2],
               [1, 23, 0, 1],
               [0, 0, 14, 1],
               [2, 3, 1, 17]])  # Replace with your confusion matrix

def calculate_specificity(cm):
    specificity_list = []

    for i in range(len(cm)):
        true_negative = np.delete(np.delete(cm, i, axis=0), i, axis=1).sum()
        false_positive = np.delete(cm, i, axis=0)[:, i].sum()
        specificity = true_negative / (true_negative + false_positive)
        specificity_list.append(specificity)

    return np.array(specificity_list)

# Usage
specificity_per_class = calculate_specificity(cm)
print(f'Specificity per class: {specificity_per_class}')

# If you want to compute the average specificity
average_specificity = np.mean(specificity_per_class)
print(f'Average Specificity: {average_specificity}')
