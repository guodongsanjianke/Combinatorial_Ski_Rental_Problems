from collections import defaultdict
import numpy as np
from tqdm import tqdm, trange
import basic_func
from basic_func import opt_line
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
import math
from functools import partial


wb = openpyxl.Workbook()

n = 6
b = [100, 95, 90, 85, 90, 75]
r = [1, 1.05, 1.10, 1.15, 1.20, 1.25]


b_dict = {}
for idx in range(1, n + 1):
    b_dict[(idx,)] = b[idx - 1]

opt_lines = [opt_line(b=0.0, r=r[0])]
opt_lines.append(opt_line(b=b[n - 1], r=0.0))

opt_result, T = basic_func.get_opt_result1(opt_lines)


shop_paths = []
for i in range(1, n + 1):
    shop_paths.append([tuple([i])])


x_list = []  # robustness
y_list = []  # consistency
path_names = []
df = pd.DataFrame()


class CSR_with_ML:
    def __init__(self, Lambda, all_partitions, b_dict, r, Ext_opt_result, T):
        self.results1_for_all_path = None
        self.results2_for_all_path = None

        self.cpt_all_path(Lambda, all_partitions, b_dict, r, Ext_opt_result, T)

    def get_all_path(self, all_partitions, b_dict, r, Ext_opt_result, l):

        if not all_partitions:
            raise ValueError("all_partitions cannot be empty")

        u_b, u_r, partitions_list = basic_func.transfer(all_partitions, b_dict, r)
        min_c_max, prob = basic_func.get_min_c_max(all_partitions, u_b, u_r, partitions_list, l, Ext_opt_result)

        t_b, t_r = basic_func.renew_b_r(u_b, u_r, partitions_list, prob)
        p_list, start_day, ALG_x = basic_func.P_list_and_get_ALG(t_b, t_r, min_c_max, l, Ext_opt_result, ret_alg=True)

        p_list_dict = defaultdict(lambda: defaultdict(str))

        extend_T = l + 5
        for i in range(len(p_list)):
            partitions_list[i].p_list = p_list[i]
            p_list_dict[partitions_list[i].p_id][partitions_list[i].split] = basic_func.p_list_extend(p_list[i],
                                                                                                      extend_T,
                                                                                                      start_day[i] - 1)
        all_path = basic_func.All_Path(n=len(all_partitions))
        for index, partition in enumerate(all_partitions):
            # Sort based on the ratio of prices to rent for product combinations
            combined_info = basic_func.combinations_sort(partition, r, b_dict)
            b_, r_, split = map(list, zip(*combined_info))
            p_list = []
            for s in split:
                p_list.append(p_list_dict[index][s])
            all_path.addPath(basic_func.path(prob[index], p_list, b_, r_, split))

        return all_path

    def cpt_all_path(self, Lambda, all_partitions, b_dict, r, Ext_opt_result, T):
        k =  math.floor(Lambda * 75)
        self.results1_for_all_path = self.get_all_path(all_partitions, b_dict, r, Ext_opt_result, k)


        l = int(np.ceil(100 / Lambda))
        self.results2_for_all_path = self.get_all_path(all_partitions, b_dict, r, Ext_opt_result, l)

    def get_result(self, case):
        if case == 1:
            return self.results1_for_all_path
        elif case == 2:
            return self.results2_for_all_path
        return None


def cr_compute(b_dict, r, Lambda, this_partitions, Ext_opt_result, opt_result, T, x, y):
    n = len(this_partitions)

    if y >= T:
        l = math.floor(Lambda * 75)
    else:
        l = int(np.ceil(100 / Lambda))
    t = min(x, l)

    if n == 1:

        min_c_max, com_order = basic_func.get_cr_one_path(this_partitions, b_dict, r, l, Ext_opt_result)
        b, r, combined_commodity = zip(*com_order)
        prob_order, _, ALG_x = basic_func.P_list_and_get_ALG(b, r, min_c_max, t, Ext_opt_result, ret_alg=True)
    else:

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


def CR_generate_multi_shop(b_dict=None, r=None):
    global x_list, y_list, path_names, df

    x_list = []
    y_list = []
    path_names = []
    df = pd.DataFrame()

    individual_partitions = [
        [shop_path] for shop_path in shop_paths
    ]


    all_partitions = shop_paths.copy()

    analyze_partitions = individual_partitions + [all_partitions]


    for path_idx, this_partition in enumerate(analyze_partitions):
        sub_robustness_list = []
        sub_consistency_list = []
        Lambda_list = []


        if path_idx < len(individual_partitions):
            path_name = f"shop_{path_idx + 1}"
        else:
            path_name = "all_shops"
        path_names.append(path_name)


        for i in trange(5, 100 + 1, 5):
            Lambda = i / 100
            Lambda_list.append(Lambda)


            Ext_opt_result = []
            k = opt_result[T] / T
            for j in range(len(opt_result) - 1):
                if opt_result[j + 1] - opt_result[j] < k:
                    Ext_opt_result = opt_result[:j + 1]
                    break
            while len(Ext_opt_result) <= math.ceil(100 / Lambda):
                if len(Ext_opt_result) > 0:
                    Ext_opt_result.append(Ext_opt_result[-1] + k)
                else:
                    Ext_opt_result.append(k)


            cr_registered = partial(
                cr_compute,
                b_dict, r, Lambda, this_partition,
                Ext_opt_result, opt_result, T
            )


            x = y = T
            consistency1 = cr_registered(x, y)

            x = T
            y = T - 1
            consistency2 = cr_registered(x, y)

            consistency = max(consistency1, consistency2)

            x = 1
            y = T
            robustness1 = cr_registered(x, y)

            x = math.ceil(b[0] / Lambda)
            y = T - 1
            robustness2 = cr_registered(x, y)

            robustness = max(robustness1, robustness2)

            sub_consistency_list.append(consistency)
            sub_robustness_list.append(robustness)

        x_list.append(sub_robustness_list)
        y_list.append(sub_consistency_list)


        if path_idx == 0:
            df["Lambda"] = Lambda_list

        df[f"robustness+{path_name}"] = sub_robustness_list
        df[f"consistency+{path_name}"] = sub_consistency_list


    df.to_excel('Fig_multi_shop_CR.xlsx', index=False)

    return Lambda_list


def Draw_CR():
    global x_list, y_list, path_names


    CR_generate_multi_shop(b_dict, r)


    plt.figure(figsize=(10, 6))


    for i in range(len(x_list) - 1):
        plt.plot(x_list[i], y_list[i], label=path_names[i], marker='o')


    plt.plot(x_list[-1], y_list[-1], label=path_names[-1], marker='*', linewidth=2)

    plt.title('Multi-Shop: Consistency vs Robustness')
    plt.xlabel('Robustness')
    plt.ylabel('Consistency')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.savefig('Fig_multi_shop_CR.png')
    plt.show()



if __name__ == "__main__":
    Draw_CR()