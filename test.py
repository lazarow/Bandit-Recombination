import numpy as np
import scipy.stats as stats

file_path = "test.txt"
data = np.loadtxt(file_path)

num_problems = 8
num_runs = 30

if data.shape[0] != num_runs or data.shape[1] != 2 * num_problems:
    raise ValueError("Unexpected data format! Ensure each row contains results for both algorithms.")

baseline = data[:, :num_problems*2:2]
your_algo = data[:, 1:num_problems*2:2]

alpha = 0.05  # Significance level
alpha_corrected = alpha / num_problems  # Bonferroni correction

results = []

for i in range(num_problems):  
    y = your_algo[:, i]
    b = baseline[:, i]
    
    differences = y - b

    stat, p_normal = stats.shapiro(differences)

    if p_normal > 0.05:
        test_stat, p_value = stats.ttest_rel(y, b)
        test_used = "Paired t-test"
    else:
        test_stat, p_value = stats.wilcoxon(y, b)
        test_used = "Wilcoxon signed-rank test"
    
    # Effect size calculations
    cohen_d = (np.mean(y) - np.mean(b)) / np.std(np.concatenate([y, b]))  
    cliff_delta = (np.sum(y > b) - np.sum(y < b)) / len(y)

    results.append((i+1, test_used, p_value, cohen_d, cliff_delta))

# Print results
print(f"{'Problem':<10}{'Test':<25}{'p-value':<12}{'Cohen\'s d':<12}{'Cliff\'s delta'}")
for res in results:
    significance = "*" if res[2] < alpha_corrected else ""  
    print(f"{res[0]:<10}{res[1]:<25}{res[2]:<12.5f}{res[3]:<12.3f}{res[4]:<12.3f} {significance}")
