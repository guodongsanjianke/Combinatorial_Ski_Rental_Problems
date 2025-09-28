import numpy as np
from CSR import csr,evaluate
from tqdm import trange
from basic_func import opt_line
from copy import deepcopy
import openpyxl
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument('--eval_T', type=int, default=None, help='Maximum number of test days')
parser.add_argument('--num_eval', type=int, default=100000, help='Number of samples per day')
args = parser.parse_args()

wb = openpyxl.Workbook()


all_values = [202, 535, 960, 370, 206, 171, 800, 120, 714, 221,
     566, 314, 430, 558, 187, 472, 199, 971, 763, 230]

n = 10
b = all_values[:n]
print(b)
r = np.ones(n)

# All product combinations with discounts
discounts = [
    ([1, 2], (b[0] + b[1]) * 0.9),
    ([2,3,4], (b[1] + b[2] + b[3]) * 0.92),
    ([4,5],(b[3] + b[4]) * 0.91),
    ([5,6],(b[4] + b[5]) * 0.87)
]


print("discounts", discounts)

opt_lines = [opt_line(b=0.0,r=n)]


def generate_replacements(sequence, combinations_to_replace):
    def is_replaceable(subseq, combination):
        """Check if it is replaceable"""
        return all((x,) in subseq for x in combination)

    def replace(subseq, combination):
        """Replace the specified combination"""
        new_seq = [x for x in subseq if x not in [(y,) for y in combination]]
        new_seq.append(tuple(combination))
        return new_seq

    # All possible replacement results
    def backtrack(subseq, remaining_combinations, results):
        if not remaining_combinations:
            results.append(subseq)
            return

        # Current combination
        current_combination = remaining_combinations[0]
        # Remaining combinations
        next_combinations = remaining_combinations[1:]

        # Do not replace the current combination
        backtrack(subseq, next_combinations, results)

        # Replace the current combination
        if is_replaceable(subseq, current_combination):
            replaced_seq = replace(subseq, current_combination)
            backtrack(replaced_seq, next_combinations, results)

    results = []
    backtrack(sequence, combinations_to_replace, results)
    return results


def min_cost(prices, discounts, m):
    n = len(prices)
    # Initialize the DP array
    max_state = 1 << n
    dp = [float('inf')] * max_state
    dp[0] = 0

    # Convert discounts into binary mask form
    discount_masks = []
    for items, cost in discounts:
        mask = sum(1 << (item - 1) for item in items)
        discount_masks.append((mask, cost))

    # Solve with dynamic programming
    for state in range(max_state):
        # Purchase individual items
        for i in range(n):
            if not (state & (1 << i)):  # The current item has not been bought
                dp[state | (1 << i)] = min(dp[state | (1 << i)], dp[state] + prices[i])

        # Apply discounts
        for mask, cost in discount_masks:
            if state & mask == 0:  # Discount items have not been fully selected
                dp[state | mask] = min(dp[state | mask], dp[state] + cost)

    # Find the minimum price for selecting m items
    min_price = float('inf')
    for state in range(max_state):
        if bin(state).count('1') == m:  # Count the number of selected items
            min_price = min(min_price, dp[state])

    return min_price



for num in trange(1,n+1):
    opt_lines.append(opt_line(b=min_cost(b,discounts,num),r=n-num))

print("opt_lines", opt_lines)


b_dict = {}
for idx in trange(1,n+1):
    b_dict[(idx,)] = b[idx-1]

for item in discounts:
    b_dict[tuple(item[0])] = item[1]

# Partition of all items purchased separately
all_split_partitions = [(i,) for i in range(1, n+1)]

# According to the discount combination Settings,
# all partitions with discounts are obtained by means of replacement
combinations_to_replace = [tuple(item[0]) for item in discounts]
all_replacements = generate_replacements(all_split_partitions, combinations_to_replace)


print("all_replacements: ", all_replacements )
fixed_partitions = deepcopy(all_replacements)

result = csr(b,r,b_dict=b_dict,opt_lines=opt_lines,fixed_partitions=fixed_partitions,eval_T=args.eval_T)

CR = result['CR']
print(f'Non-upgrading，CR：{CR}')


print('########################')
print('Non-upgrading, the probability of 2 layers')

all_path,opt_result,eval_T = result['all_path'],result['opt_result'],result['eval_T']
sheet1 = wb.active
sheet1.title = "Non-upgrading"
sheet1.append(["Partitions of Items", "The probability of the first layer", "The probability of the second layer"])


for path in all_path.paths:
    print(str(path.items_split)+":")
    print(f'The probability of the first layer:{path.prob}')
    sheet1.append([str(path.items_split), path.prob, str(path.p_list)])



eval_cr = evaluate(all_path, opt_result, eval_T,num_evaluate=args.num_eval)
sheet1.append(["CR",f"{CR}","evaluate_cr",str(eval_cr)])

wb.save("10_item.xlsx")
print("Data has been successfully written to 10_item.xlsx")
