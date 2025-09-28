import numpy as np
from CSR import csr, evaluate
from tqdm import trange
from basic_func import opt_line, All_Path, path, get_cr_one_path, P_list_and_get_ALG, combinations_sort, transfer, get_min_c_max, renew_b_r, p_list_extend, sample_evaluate, opt_result_extend
from copy import deepcopy
import openpyxl
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import random
from collections import defaultdict
import math
import os
import torch
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

parser = argparse.ArgumentParser(description="")
parser.add_argument('--eval_T', type=int, default=None, help='Maximum number of test days')
parser.add_argument('--num_eval', type=int, default=10000, help='Number of samples per day')
parser.add_argument('--use_gpu', action='store_true', help='Accelerated computation using GPUs')
parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for parallel processing')
args = parser.parse_args()


use_gpu = args.use_gpu and torch.cuda.is_available()
if use_gpu:
    print(f"Using the GPU: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda:0")
else:
    print("Using the CPU")
    device = torch.device("cpu")

wb = openpyxl.Workbook()

all_values = [202, 535, 960, 370, 206, 171, 800, 120, 714, 221,
              566, 314, 430, 558, 187, 472, 199, 971, 763, 230]

n = 10
b = all_values[:n]
r = np.ones(n)

discounts = [
    ([1, 2], (b[0] + b[1]) * 0.9),
    ([2, 3, 4], (b[1] + b[2] + b[3]) * 0.92),
    ([4, 5], (b[3] + b[4]) * 0.91),
    ([5, 6], (b[4] + b[5]) * 0.87)
]

opt_lines = [opt_line(b=0.0, r=n)]

def generate_replacements(sequence, combinations_to_replace):
    def is_replaceable(subseq, combination):
        return all((x,) in subseq for x in combination)
    def replace(subseq, combination):
        new_seq = [x for x in subseq if x not in [(y,) for y in combination]]
        new_seq.append(tuple(combination))
        return new_seq
    def backtrack(subseq, remaining_combinations, results):
        if not remaining_combinations:
            results.append(subseq)
            return
        current_combination = remaining_combinations[0]
        next_combinations = remaining_combinations[1:]
        backtrack(subseq, next_combinations, results)
        if is_replaceable(subseq, current_combination):
            replaced_seq = replace(subseq, current_combination)
            backtrack(replaced_seq, next_combinations, results)
    results = []
    backtrack(sequence, combinations_to_replace, results)
    return results

def min_cost(prices, discounts, m):
    n = len(prices)
    max_state = 1 << n
    dp = [float('inf')] * max_state
    dp[0] = 0
    discount_masks = []
    for items, cost in discounts:
        mask = sum(1 << (item - 1) for item in items)
        discount_masks.append((mask, cost))
    for state in range(max_state):
        for i in range(n):
            if not (state & (1 << i)):
                dp[state | (1 << i)] = min(dp[state | (1 << i)], dp[state] + prices[i])
        for mask, cost in discount_masks:
            if state & mask == 0:
                dp[state | mask] = min(dp[state | mask], dp[state] + cost)
    min_price = float('inf')
    for state in range(max_state):
        if bin(state).count('1') == m:
            min_price = min(min_price, dp[state])
    return min_price

for num in trange(1, n + 1):
    opt_lines.append(opt_line(b=min_cost(b, discounts, num), r=n - num))

b_dict = {}
for idx in trange(1, n + 1):
    b_dict[(idx,)] = b[idx - 1]
for item in discounts:
    b_dict[tuple(item[0])] = item[1]

all_split_partitions = [(i,) for i in range(1, n + 1)]
combinations_to_replace = [tuple(item[0]) for item in discounts]
all_replacements = generate_replacements(all_split_partitions, combinations_to_replace)
fixed_partitions = deepcopy(all_replacements)

x_list = []
y_list = []
order_list = []
df = pd.DataFrame()

Lambda = 0.25
num_eval = args.num_eval

def y_sampling(x, std, batch_size=None):
    """
    Sample y-values using GPU acceleration

    If batch_size is None, return a single sample
    Otherwise return an array of samples of size batch_size
    """
    if batch_size is None:
        if use_gpu:

            epsilon = torch.normal(0, std, size=(1,), device=device).item()
        else:
            epsilon = np.random.normal(loc=0, scale=std)
        return x + epsilon
    else:
        if use_gpu:
            # Generating a batch of random numbers on the GPU using PyTorch
            epsilon = torch.normal(0, std, size=(batch_size,), device=device).cpu().numpy()
        else:
            epsilon = np.random.normal(loc=0, scale=std, size=batch_size)
        return x + epsilon

def precompute_case_data(b_dict, r, Lambda, this_partition, Ext_opt_result, T, case):
    n = len(this_partition)
    if case == 1:
        l = math.floor(Lambda * T)
    else:
        l = math.ceil(T / Lambda)
    if n == 1:
        min_c_max, com_order = get_cr_one_path(this_partition, b_dict, r, l, Ext_opt_result)
        t_b, t_r, combined_commodity = zip(*com_order)
        p_list, start_days = P_list_and_get_ALG(t_b, t_r, min_c_max, l, Ext_opt_result)
        combined = combinations_sort(this_partition[0], r, b_dict)
        total_buy, total_rent, total_commodity_order = zip(*combined)
        return {
            'p_list': p_list,
            'start_days': start_days,
            'combined': combined,
            'total_buy': total_buy,
            'total_rent': total_rent
        }
    else:
        u_b, u_r, partitions_list = transfer(this_partition, b_dict, r)
        min_c_max, prob = get_min_c_max(this_partition, u_b, u_r, partitions_list, l, Ext_opt_result)
        t_b, t_r = renew_b_r(u_b, u_r, partitions_list, prob)
        p_list, start_day, _ = P_list_and_get_ALG(t_b, t_r, min_c_max, l, Ext_opt_result, ret_alg=True)
        p_list_dict = defaultdict(lambda: defaultdict(str))
        extend_T = l + 5
        for i in range(len(p_list)):
            partitions_list[i].p_list = p_list[i]
            p_list_dict[partitions_list[i].p_id][partitions_list[i].split] = p_list_extend(p_list[i], extend_T, start_day[i] - 1)
        all_path = All_Path(n=len(this_partition))
        for index, partition in enumerate(this_partition):
            combined_info = combinations_sort(partition, r, b_dict)
            b_, r_, split = map(list, zip(*combined_info))
            path_p_list = []
            for s in split:
                path_p_list.append(p_list_dict[index][s])
            all_path.addPath(path(prob[index], path_p_list, b_, r_, split))
        return {
            'all_path': all_path
        }

def cr_compute_with_precomputed(precomputed_data, n, opt_result, T, x, y):
    if y >= T:
        case_data = precomputed_data[1]
    else:
        case_data = precomputed_data[2]
    if n == 1:
        p_list = case_data['p_list']
        start_days = case_data['start_days']
        total_buy = case_data['total_buy']
        total_rent = case_data['total_rent']
        buy_days = []
        for i in range(len(p_list)):
            day_list = [i + 1 for i in range(len(p_list[i]))]
            buy_day = random.choices(day_list, p_list[i], k=1)
            buy_days.append(buy_day[0] + start_days[i] - 1)
        ALG_x = 0
        for i in range(len(p_list)):
            if x >= buy_days[i]:
                ALG_x += total_rent[i] * (buy_days[i] - 1)
                ALG_x += total_buy[i]
            else:
                ALG_x += total_rent[i] * x
    else:
        all_path = case_data['all_path']
        ALG_x = sample_evaluate(all_path, opt_result, end_T=x, ret_cr=False, ret_alg=True)
    if x >= T:
        OPT_x = opt_result[T]
    else:
        OPT_x = opt_result[x]
    CR = ALG_x / OPT_x
    return CR


def batch_cr_compute(precomputed_data, n, opt_result, eval_T, x_batch, y_batch):
    results = []
    for x, y in zip(x_batch, y_batch):
        cr = cr_compute_with_precomputed(precomputed_data, n, opt_result, eval_T, x, y)
        results.append(cr)
    return results


def gpu_accelerated_compute(data, function_type):

    if not use_gpu or not isinstance(data, (list, np.ndarray)):

        if function_type == 'mean':
            return np.mean(data)
        elif function_type == 'sum':
            return np.sum(data)
        elif function_type == 'max':
            return np.max(data)
        elif function_type == 'min':
            return np.min(data)
        else:
            return data
    

    tensor_data = torch.tensor(data, device=device)
    

    if function_type == 'mean':
        result = torch.mean(tensor_data).item()
    elif function_type == 'sum':
        result = torch.sum(tensor_data).item()
    elif function_type == 'max':
        result = torch.max(tensor_data).item()
    elif function_type == 'min':
        result = torch.min(tensor_data).item()
    else:

        result = tensor_data.cpu().numpy()
        
    return result


def process_samples_in_parallel(samples, precomputed_data, n, opt_result, eval_T, std, batch_size=100):
    num_samples = len(samples)
    results = []
    

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for i in range(0, num_samples, batch_size):
            x_batch = samples[i:i+batch_size]
            

            if use_gpu:
                # Convert x_batch to tensor for batch operations
                x_tensor = torch.tensor(x_batch, device=device, dtype=torch.float32)
                # Generate normally distributed random numbers of corresponding size
                epsilon = torch.normal(0, std, size=x_tensor.size(), device=device)
                # Calculate y = x + epsilon
                y_tensor = x_tensor + epsilon
                # Turn back to CPU and turn to Python lists
                y_batch = y_tensor.cpu().numpy().tolist()
            else:
                # Batch computing on CPUs
                y_batch = [x + np.random.normal(0, std) for x in x_batch]
            
            # Submission of batch tasks
            future = executor.submit(batch_cr_compute, precomputed_data, n, opt_result, eval_T, x_batch, y_batch)
            futures.append(future)
        
        # Collect all results
        for future in futures:
            batch_results = future.result()
            results.extend(batch_results)
    
    return results

result = csr(b, r, b_dict=b_dict, opt_lines=opt_lines, fixed_partitions=fixed_partitions, eval_T=args.eval_T)
CR = result['CR']
all_path, opt_result, eval_T = result['all_path'], result['opt_result'], result['eval_T']

# Validate and extend opt_result to ensure eval_T is within bounds
if eval_T >= len(opt_result):
    opt_result = opt_result_extend(opt_result, eval_T + 1)
    print(f"Extended opt_result to length {len(opt_result)} to accommodate eval_T={eval_T}")

s = opt_result[eval_T] / eval_T
Ext_opt_result = []
for i in range(len(opt_result) - 1):
    if opt_result[i + 1] - opt_result[i] < s:
        Ext_opt_result = opt_result[:i + 1]
        break
max_l = math.ceil(eval_T / Lambda)
while len(Ext_opt_result) <= max_l:
    if len(Ext_opt_result) > 0:
        Ext_opt_result.append(Ext_opt_result[-1] + s)
    else:
        Ext_opt_result.append(s)

iter_partitions = fixed_partitions + [fixed_partitions]
precomputed_data = {}
for idx, partition in enumerate(iter_partitions):
    this_partition = [partition] if idx < len(fixed_partitions) else partition
    precomputed_data[idx] = {
        1: precompute_case_data(b_dict, r, Lambda, this_partition, Ext_opt_result, eval_T, 1),
        2: precompute_case_data(b_dict, r, Lambda, this_partition, Ext_opt_result, eval_T, 2)
    }
    order_list.append(str(this_partition) if idx < len(fixed_partitions) else "best")

for i in range(len(fixed_partitions) + 1):
    y_list.append([])


print("Pre-generate all sampling points...")
if use_gpu:

    x_values = torch.randint(1, 4 * eval_T + 1, (num_eval,), device=device).cpu().numpy()
    sample_points = x_values.tolist()
else:
    sample_points = [random.randint(1, 4 * eval_T) for _ in trange(num_eval)]


for STD in trange(3 * eval_T):
    x_list.append(STD)
    
    for idx, partition in enumerate(iter_partitions):
        this_partition = [partition] if idx < len(fixed_partitions) else partition
        n = len(this_partition)
        

        c_list = process_samples_in_parallel(
            sample_points, 
            precomputed_data[idx], 
            n, 
            opt_result, 
            eval_T, 
            STD,
            batch_size=min(1000, num_eval)
        )
        

        if use_gpu and len(c_list) > 0:
            mean_cr = torch.tensor(c_list, device=device).mean().item()
        else:
            mean_cr = np.mean(c_list)
            
        y_list[idx].append(mean_cr)
        
    if (STD + 1) % 100 == 0:
        os.makedirs("./Intermediate results", exist_ok=True)
        plt.figure()
        mpl.rcParams['font.family'] = 'SimHei'
        plt.rcParams['axes.unicode_minus'] = False
        for i in range(len(y_list) - 1):
            plt.plot(x_list, y_list[i], label=f"{order_list[i]}")
        plt.plot(x_list, y_list[-1], label="best")
        plt.title('CR vs STD')
        plt.xlabel('STD')
        plt.ylabel('CR')
        plt.legend()
        plt.savefig(f'./Intermediate results/CR_STD_[x: 0-{STD} ].png')
        plt.close()


if use_gpu:
    torch.cuda.empty_cache()

df['x(STD)'] = x_list
for idx, partition in enumerate(fixed_partitions):
    df[f'{order_list[idx]}'] = y_list[idx]
df['best'] = y_list[-1]
df.to_excel('CR_STD.xlsx', index=False)

sheet1 = wb.active
sheet1.title = "Non-upgrading"
sheet1.append(["Partitions of Items", "The probability of the first layer", "The probability of the second layer"])
for path in all_path.paths:
    sheet1.append([str(path.items_split), path.prob, str(path.p_list)])
eval_cr = evaluate(all_path, opt_result, eval_T, num_evaluate=args.num_eval)
sheet1.append(["CR", f"{CR}", "evaluate_cr", str(eval_cr)])
wb.save("10_item.xlsx")

mpl.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
for i in range(len(y_list) - 1):
    plt.plot(x_list, y_list[i], label=f"{order_list[i]}")
plt.plot(x_list, y_list[-1], label="best")
plt.title('CR vs STD')
plt.xlabel('STD')
plt.ylabel('CR')
plt.legend()
plt.savefig('CR_STD.png')
plt.close()