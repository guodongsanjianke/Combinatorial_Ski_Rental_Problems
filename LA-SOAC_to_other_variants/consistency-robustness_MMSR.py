import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

lambda_list = []
r_x_list = []
c_y_list = []

def cpt_cr(lumda, x, y, b1, b2, r1, r2):
    p = np.random.rand()
    k = math.floor(lumda * 75)
    l = int(np.ceil(100 / lumda))

    if x >= b2:
        opt = b2
    else:
        opt = x

    if y >= b2:
        # Calculate qs
        q = [((b2 - r2) / b2) ** (k - i) * (r2 / (b2 * (1 - (1 - r2 / b2) ** k))) for i in range(1, k + 1)]
        qs = np.cumsum(q)

        cost = 0
        for day in range(min(x, k)):
            cost += b2 * q[day] + r2 * (1 - qs[day])
    else:
        # Calculate rs
        r = [((b1 - 1) / b1) ** (l - i) * (1 / (b1 * (1 - (1 - 1 / b1) ** l))) for i in range(1, l + 1)]
        rs = np.cumsum(r)

        cost = 0
        for day in range(min(x, l)):
            cost += b1 * r[day] + r1 * (1 - rs[day])

    cr = cost / opt
    return cr

# Parameters
r1, b1 = 1, 100
r2, b2 = 1.25, 75

# Calculate robustness and consistency for each lambda
for i in tqdm(range(5, 100 + 1, 5)):
    lumda = i / 100
    lambda_list.append(lumda)

    x = y = b2
    consistency1 = cpt_cr(lumda, x, y, b1, b2, r1, r2)

    x = b2
    y = b2 - 1
    consistency2 = cpt_cr(lumda, x, y, b1, b2, r1, r2)

    consistency = max(consistency1, consistency2)

    x = 1
    y = b2
    robustness1 = cpt_cr(lumda, x, y, b1, b2, r1, r2)

    x = math.ceil(b1 / lumda)
    y = b2 - 1
    robustness2 = cpt_cr(lumda, x, y, b1, b2, r1, r2)

    robustness = max(robustness1, robustness2)

    r_x_list.append(robustness)
    c_y_list.append(consistency)

# Create a DataFrame
data = {
    'lambda': lambda_list,
    'robustness': r_x_list,
    'consistency': c_y_list
}
df = pd.DataFrame(data)

# Save to Excel
df.to_excel('robustness_consistency_results.xlsx', index=False)

print("Data successfully saved to 'robustness_consistency_results.xlsx'")

# Also plot the data
plt.figure(figsize=(10, 6))
plt.plot(r_x_list, c_y_list, 'o-', label='multi_shop')
plt.title('Fig5-p1')
plt.xlabel('robustness')
plt.ylabel('consistency')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('Fig5-p1.png')
plt.show()