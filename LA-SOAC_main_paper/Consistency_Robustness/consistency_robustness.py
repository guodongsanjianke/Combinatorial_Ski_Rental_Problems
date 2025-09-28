import copy

import numpy as np
import basic_func
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from tqdm import tqdm
from functools import partial

x_list = []
y_list = []
order_list = []
y_upper_bound = []
y_lower_bound = []

all_partitions = []
b_list = []
count_delta = 0
df = pd.DataFrame()

# b = [149.99, 149.99, 149.99]
r = [1.0,1.0,1.0]

# b_12 = 229.99
# b_13 = 229.99
# b_23 = 229.99
# b_123 = 329.99

# Define the price of a combination of items
b_dict = {
    (): 0,
    (1,): 80,
    (2,): 110,
    (3,): 130,
    (1, 2): 180.5,
    (1, 3): 199.5,
    (2, 3): 228,
    (1, 2, 3): 288,
}


def y_sampling(x, std):
    epsilon = np.random.normal(loc=0, scale=std)
    y = x + epsilon
    return y


def cr_compute(b_dict, r, Lambda, this_partitions, Ext_opt_result, opt_result, T, x, y):
    n = len(this_partitions)

    if y >= T:
        l = math.floor(Lambda * T)
    else:
        l = math.ceil(T / Lambda)

    t = min(x, l)

    if n == 1:
        # one path
        min_c_max, com_order = basic_func.get_cr_one_path(this_partitions, b_dict, r, l, Ext_opt_result)
        b, r, combined_commodity = zip(*com_order)
        prob_order, _, ALG_x = basic_func.P_list_and_get_ALG(b, r, min_c_max, t, Ext_opt_result, ret_alg=True)
    else:
        # all paths , == CSR
        u_b, u_r, partitions_list = basic_func.transfer(this_partitions, b_dict, r)
        min_c_max, prob = basic_func.get_min_c_max(this_partitions, u_b, u_r, partitions_list, l, Ext_opt_result)
        t_b, t_r = basic_func.renew_b_r(u_b, u_r, partitions_list, prob)
        p_list, start_day, ALG_x = basic_func.P_list_and_get_ALG(t_b, t_r, min_c_max, t, Ext_opt_result, ret_alg=True)

    if x >= T:
        OPT_x = opt_result[T]
    else:
        OPT_x = opt_result[x]

    CR = ALG_x / OPT_x

    return CR


def CR_generate(b_dict=None, r=None, discount=0.05):
    global all_partitions, b_list

    n = 3
    product_list = list(range(1, n + 1))
    two_partitions = list(basic_func.get_two_partitions(product_list))

    if r is None:
        r = []
        for i in range(n):
            r.append(1)

    if b_dict is None:
        b_dict = {}
        b_list = [80, 110, 130]
        for partition in two_partitions:
            b_dict[tuple(partition[0])] = sum([b_list[i - 1] for i in partition[0]]) * (
                    1 - discount * (len(partition[0]) - 1))

    all_partitions = basic_func.get_all_partitions(product_list[::-1])

    # temp = []
    # for partition in all_partitions:
    #     temp.append(partition)
    # # all_partitions.append(temp)
    temp = copy.deepcopy(all_partitions)
    iter_partitions = all_partitions + [temp]

    opt_result, T, _ = basic_func.get_opt_result(b_dict, r, two_partitions, product_list)

    for partition in iter_partitions:
        sub_y_list = []
        sub_x_list = []
        Lambda_list = []
        if not partition == all_partitions:
            this_partition = [partition]
        else:
            this_partition = partition

        for i in tqdm(range(20, 100 + 1, 5)):
            Lambda = i / 100
            Lambda_list.append(Lambda)

            # Extend the opt_result
            Ext_opt_result = []
            k = opt_result[T] / T
            for i in range(len(opt_result) - 1):
                if opt_result[i + 1] - opt_result[i] < k:
                    Ext_opt_result = opt_result[:i + 1]
                    break
            while len(Ext_opt_result) <= math.ceil(T / Lambda):
                Ext_opt_result.append(Ext_opt_result[-1] + k)

            # opt_result_extend = basic_func.opt_result_extend(opt_result, extend_T)
            # Get the value of consistency and robustness when y≥T or y<T

            cr_registered = partial(
                cr_compute,
                b_dict, r, Lambda, this_partition,
                Ext_opt_result, opt_result, T
            )  # register together params

            x = y = T
            consistency1 = cr_registered(x, y)

            x = T
            y = T - 1
            consistency2 = cr_registered(x, y)

            consistency = max(consistency1, consistency2)

            # x = 1
            # y = T
            x = math.floor(Lambda * T)
            y = T
            robustness1 = cr_registered(x, y)

            x = math.ceil(T / Lambda)
            y = T - 1
            robustness2 = cr_registered(x, y)

            robustness = max(robustness1, robustness2)

            sub_x_list.append(robustness)
            sub_y_list.append(consistency)

        if not partition == all_partitions:
            order_list.append(basic_func.get_order(partition, b_dict, r))
            name = f'{order_list[-1]}'
        else:
            name = 'best'

        if partition == all_partitions[0]:
            df[name] = Lambda_list
        else:
            df[name] = ''

        df[f'robustness+'+name] = sub_x_list
        df[f'consistency+'+name] = sub_y_list

        x_list.append(sub_x_list)
        y_list.append(sub_y_list)

    df.to_excel('Fig5-part1.xlsx', index=False)


def Draw():
    mpl.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False

    for i in range(len(x_list) - 1):
        plt.plot(x_list[i], y_list[i], label=f'{order_list[i]}', marker='o')
    plt.plot(x_list[-1], y_list[-1], label='best', marker='*')

    plt.title(f'Fig5-p1')
    plt.xlabel('robustness')
    plt.ylabel('consistency')

    plt.legend()

    plt.savefig('Fig5-p1.png')

    plt.show()


if __name__ == "__main__":
    CR_generate(b_dict, r)
    Draw()
