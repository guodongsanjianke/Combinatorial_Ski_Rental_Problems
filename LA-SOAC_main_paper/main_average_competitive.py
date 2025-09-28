import os

import numpy as np
from tqdm import trange
from basic_func import All_Path, path
import basic_func
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import random
from collections import defaultdict
import copy

x_list = []
y_list = []
order_list = []
df = pd.DataFrame()

Lambda = 0.25
num_eval = 10000


def y_sampling(x, std):
    epsilon = np.random.normal(loc=0, scale=std)
    y = x + epsilon

    return y


def precompute_case_data(b_dict, r, Lambda, this_partition, Ext_opt_result, T, case):
    """
    case = 1: y >= T
    case = 2: y < T
    """
    n = len(this_partition)

    if case == 1:  # y >= T
        l = math.floor(Lambda * T)
    else:  # y < T
        l = math.ceil(T / Lambda)

    if n == 1:
        min_c_max, com_order = basic_func.get_cr_one_path(this_partition, b_dict, r, l, Ext_opt_result)
        t_b, t_r, combined_commodity = zip(*com_order)
        p_list, start_days = basic_func.P_list_and_get_ALG(t_b, t_r, min_c_max, l, Ext_opt_result)

        combined = basic_func.combinations_sort(this_partition[0], r, b_dict)
        total_buy, total_rent, total_commodity_order = zip(*combined)

        return {
            'p_list': p_list,
            'start_days': start_days,
            'combined': combined,
            'total_buy': total_buy,
            'total_rent': total_rent
        }
    else:
        u_b, u_r, partitions_list = basic_func.transfer(this_partition, b_dict, r)
        min_c_max, prob = basic_func.get_min_c_max(this_partition, u_b, u_r, partitions_list, l, Ext_opt_result)
        t_b, t_r = basic_func.renew_b_r(u_b, u_r, partitions_list, prob)
        p_list, start_day, _ = basic_func.P_list_and_get_ALG(t_b, t_r, min_c_max, l, Ext_opt_result, ret_alg=True)

        p_list_dict = defaultdict(lambda: defaultdict(str))

        extend_T = l + 5
        for i in range(len(p_list)):
            partitions_list[i].p_list = p_list[i]
            p_list_dict[partitions_list[i].p_id][partitions_list[i].split] = basic_func.p_list_extend(p_list[i],
                                                                                                      extend_T,
                                                                                                      start_day[i] - 1)

        all_path = All_Path(n=len(this_partition))
        for index, partition in enumerate(this_partition):
            # Sort based on the ratio of prices to rent for product combinations
            combined_info = basic_func.combinations_sort(partition, r, b_dict)
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

        # buy_days is the time of purchase for each combination
        # which is randomly selected according to the p_list
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
        ALG_x = basic_func.sample_evaluate(all_path, opt_result, end_T=x, ret_cr=False, ret_alg=True)

    if x >= T:
        OPT_x = opt_result[T]
    else:
        OPT_x = opt_result[x]

    CR = ALG_x / OPT_x

    return CR


def CR_generate(b_dict=None, r=None, discount=0.05, n=3):
    global count_delta, b_list, x_list, y_list

    product_list = list(range(1, n + 1))
    two_partitions = list(basic_func.get_two_partitions(product_list))

    if r is None:
        r = [1] * n

    if b_dict is None:
        b_dict = {}
        b_list = [80, 110, 130]
        for partition in two_partitions:
            b_dict[tuple(partition[0])] = sum([b_list[i - 1] for i in partition[0]]) * (
                    1 - discount * (len(partition[0]) - 1))

    # b_ is the price of all goods purchased as a combination
    b_ = int(b_dict[tuple(product_list)])
    all_partitions = basic_func.get_all_partitions(product_list[::-1])
    temp = copy.deepcopy(all_partitions)
    iter_partitions = all_partitions + [temp]

    opt_result, T, _ = basic_func.get_opt_result(b_dict, r, two_partitions, product_list)

    for i in range(len(all_partitions) + 1):
        y_list.append([])

    for idx, partition in enumerate(all_partitions):
        order_list.append(basic_func.get_order(partition, b_dict, r))

    s = opt_result[T] / T
    Ext_opt_result = []
    for i in range(len(opt_result) - 1):
        if opt_result[i + 1] - opt_result[i] < s:
            Ext_opt_result = opt_result[:i + 1]
            break
    max_l = math.ceil(T / Lambda)
    while len(Ext_opt_result) <= max_l:
        if len(Ext_opt_result) > 0:
            Ext_opt_result.append(Ext_opt_result[-1] + s)
        else:
            Ext_opt_result.append(s)


    precomputed_data = {}
    for idx, partition in enumerate(iter_partitions):
        this_partition = [partition] if idx < len(all_partitions) else partition
        n = len(this_partition)

        precomputed_data[idx] = {
            1: precompute_case_data(b_dict, r, Lambda, this_partition, Ext_opt_result, T, 1),
            2: precompute_case_data(b_dict, r, Lambda, this_partition, Ext_opt_result, T, 2)
        }

    for STD in trange(6 * T):
        x_list.append(STD)

        for idx, partition in enumerate(iter_partitions):
            this_partition = [partition] if idx < len(all_partitions) else partition
            n = len(this_partition)
            c_list = []

            for num in range(num_eval):
                x = random.randint(1, 4 * T)
                y = y_sampling(x, STD)

                CR = cr_compute_with_precomputed(precomputed_data[idx], n, opt_result, T, x, y)
                c_list.append(CR)

            y_list[idx].append(np.mean(c_list))


        if (STD + 1) % 100 == 0:
            os.makedirs("./Intermediate results", exist_ok=True)
            plt.figure()
            mpl.rcParams['font.family'] = 'SimHei'
            plt.rcParams['axes.unicode_minus'] = False
            for i in range(len(y_list) - 1):
                plt.plot(x_list, y_list[i], label=f"{order_list[i]}")
            plt.plot(x_list, y_list[-1], label="best")
            plt.title(f'Fig4')
            plt.xlabel('STD')
            plt.ylabel('CR')
            plt.legend()
            plt.savefig(f'./Intermediate results/Fig4_[x: 0-{STD} ].png')
            plt.close()

    df['x(STD)'] = x_list
    for idx, partition in enumerate(all_partitions):
        df[f'{order_list[idx]}'] = y_list[idx]

    df['best'] = y_list[-1]

    df.to_excel('Fig4.xlsx', index=False)


def Draw():
    mpl.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False

    for i in range(len(y_list) - 1):
        plt.plot(x_list, y_list[i], label=f"{order_list[i]}")
    plt.plot(x_list, y_list[-1], label="best")

    plt.title(f'Fig4')
    plt.xlabel('STD')
    plt.ylabel('CR')

    plt.legend()

    plt.savefig('Fig4.png')

    plt.show()


CR_generate()
Draw()