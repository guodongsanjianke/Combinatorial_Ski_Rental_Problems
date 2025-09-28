from copy import deepcopy
import numpy as np
from dataclasses import dataclass
import random


random.seed(42)
np.random.seed(42)

@dataclass
class Group:
    p_list: list
    p_id: int
    split: tuple

@dataclass
class path:
    prob: float
    p_list: list
    b: list
    r: list
    items_split: list

@dataclass
class opt_line:
    b: float
    r: float

class All_Path:
    def __init__(self, n):
        self.path_num = n
        self.paths = []
        self.path_prob = []

    def addPath(self, path):
        self.paths.append(path)
        self.path_prob.append(path.prob)

    def getPath(self, idx):
        return self.paths[idx]

E = 1e-6

def get_all_partitions(s):
    if not s:
        return [[]]
    first_element = next(iter(s))
    rest_elements = list(set(s).difference({first_element}))[::-1]
    rest_partitions = get_all_partitions(rest_elements)
    partitions = []
    for partition in rest_partitions:
        partitions.append([(first_element,)] + partition)
        for i, subset in enumerate(partition):
            new_partition = [tuple(subset) for subset in partition]
            new_partition[i] += (first_element,)
            partitions.append(new_partition)
    return partitions

def get_two_partitions(s):
    if not s:
        yield [], []
    else:
        item = s[0]
        for subset1, subset2 in get_two_partitions(s[1:]):
            yield [item] + subset1, subset2
            yield subset1, [item] + subset2

def get_opt_result(b_dict, r, two_partitions, commodity_set, b=None, n=None):
    def optimal_function(t, b_dict, r, two_partitions):
        cost_set = []
        for i in range(len(two_partitions)):
            r_cost = 0
            b_cost = b_dict[tuple(two_partitions[i][0])]
            for k in two_partitions[i][1]:
                r_cost = r_cost + r[k - 1] * t
            cost_set.append(b_cost + r_cost)
        OPT_i = cost_set.index(min(cost_set))
        OPT_cost = cost_set[OPT_i]
        return OPT_i, OPT_cost
    T = 1
    OPT_cost_ = [0]
    OPT_group_ = [None]
    while True:
        opt_i, opt_cost = optimal_function(T, b_dict, r, two_partitions)
        OPT_cost_.append(opt_cost)
        temp = tuple(two_partitions[opt_i][0])
        if temp != OPT_group_[-1] and temp != ():
            OPT_group_.append(temp)
        if opt_cost == b_dict[tuple(commodity_set)]:
            break
        T += 1
    return OPT_cost_, T, OPT_group_[1:]

def get_opt_result1(opt_lines):
    def optimal_function(t, opt_lines):
        cost_set = []
        for i, line in enumerate(opt_lines):
            b_cost = line.b
            r_cost = line.r * t
            cost_set.append(b_cost + r_cost)
        OPT_i = cost_set.index(min(cost_set))
        OPT_cost = cost_set[OPT_i]
        return OPT_i, OPT_cost
    T = 1
    OPT_cost_ = [0]
    OPT_group_ = [None]
    while True:
        opt_i, opt_cost = optimal_function(T, opt_lines)
        OPT_cost_.append(opt_cost)
        if opt_cost == opt_lines[-1].b:
            break
        T += 1
    return OPT_cost_, T

class Equ(Exception):
    pass

class Over(Exception):
    pass

def get_CR(b, r, T, opt_result):
    c_max = max(max(b), 10)
    c_min = 1
    c = 2
    epsilon = E
    while (c_max - c_min) > epsilon:
        P_i = 0
        count = 0
        for i in range(1, T + 1):
            OPT_i = opt_result[i] - opt_result[i - 1]
            c_OPT_i = c * OPT_i
            flag = True
            try:
                while flag:
                    flag = False
                    if b[count] - r[count] != 0:
                        X_i = (c_OPT_i - (1 - P_i) * r[count] - sum(r[(count + 1):])) / (b[count] - r[count])
                        P_i = P_i + X_i
                    else:
                        X_i = 1 - P_i
                        P_i = 2
                    if P_i >= 1:
                        if count == len(b) - 1:
                            if P_i == 1 and i == T:
                                return c
                            c_max = c
                            c = (c_max + c_min) / 2
                            raise Over()
                        else:
                            if P_i > 1:
                                P_i = P_i - X_i
                                c_OPT_i = c_OPT_i - (1 - P_i) * b[count]
                                P_i = X_i = 0
                                count += 1
                                flag = True
                            elif P_i == 1:
                                P_i = 0
                                count += 1
                                if i == T:
                                    c_min = c
                                    c = (c_max + c_min) / 2
                                raise Equ()
                    else:
                        if i == T:
                            c_min = c
                            c = (c_max + c_min) / 2
                            raise Over
                        elif X_i < 0:
                            c_min = c
                            c = (c_max + c_min) / 2
                            raise Over
                        else:
                            i = i + 1
            except Over:
                break
            except Equ:
                continue
    return c_max

def combinations_sort(partition, r, b_dict):
    total_r = []
    total_b = []
    for s in partition:
        rental = 0
        if isinstance(s, Group):
            s = s.split
        for one in s:
            rental += r[int(one) - 1]
        total_r.append(rental)
        total_b.append(b_dict[s])
    combined = list(zip(total_b, total_r, partition))
    combined.sort(key=lambda x: x[0] / x[1] if x[1] != 0 else float('inf'))
    return combined

def merge_and_duplicate(lists):
    partitions_list = []
    for index, split in enumerate(lists):
        for element in split:
            partitions_list.append(Group([], index, element))
    return partitions_list

def transfer(all_partitions, b_dict, r):
    partitions_list = merge_and_duplicate(all_partitions)
    combine_info = combinations_sort(partitions_list, r, b_dict)
    total_b, total_r, partitions_list = zip(*combine_info)
    total_b = list(total_b)
    total_r = list(total_r)
    partitions_list = list(partitions_list)
    return total_b, total_r, partitions_list

def transfer_for_upgrading(opt_group, b_dict, r):
    partition_list = []
    already_buy = ()
    total_b = []
    total_r = []
    for index, group in enumerate(opt_group):
        total_b.append(b_dict[group])
        total_r.append(sum([r[item - 1] for item in group]))
    for index, group in enumerate(opt_group):
        partition_list.append(Group([], -1, group))
        if index > 0:
            common_elements = tuple(set(group) & set(already_buy))
            total_b[index] = total_b[index] - b_dict[common_elements]
            total_r[index] = total_r[index] - sum([r[item - 1] for item in common_elements])
        already_buy = tuple(set(already_buy) | set(group))
    return total_b, total_r, partition_list

def renew_b_r(b, r, partitions_list, p_partition):
    b_ = deepcopy(b)
    r_ = deepcopy(r)
    for i in range(len(b_)):
        b_[i] *= p_partition[partitions_list[i].p_id]
        r_[i] *= p_partition[partitions_list[i].p_id]
    return b_, r_

def project_simplex(v):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond].max()
    theta = cssv[rho - 1] / rho
    return np.maximum(v - theta, 0)

def get_min_c_max(all_partitions, total_b, total_r, partitions_list, T, opt_result, max_epochs=6000, delta_p=2e-2, learning_rate=0.1, tol=E):
    def one_cpt(b, r, p_partition):
        b_, r_ = renew_b_r(b, r, partitions_list, p_partition)
        value = get_CR(b_, r_, T, opt_result)
        return value
    partition_len_ = len(all_partitions)
    P_partition_ = np.array([1 / partition_len_ for _ in range(partition_len_)])
    rest1 = 10
    rest2 = one_cpt(total_b, total_r, P_partition_)
    for epoch in range(max_epochs):
        grad = np.zeros_like(P_partition_)
        for i in range(len(P_partition_)):
            temp_P = deepcopy(P_partition_)
            temp_P[i] += delta_p
            temp_P = project_simplex(temp_P)
            c_temp = one_cpt(total_b, total_r, temp_P)
            grad[i] = (c_temp - rest2) / delta_p
        P_partition_ -= learning_rate * grad
        P_partition_ = project_simplex(P_partition_)
        rest1 = rest2
        rest2 = one_cpt(total_b, total_r, P_partition_)
        if abs(rest1 - rest2) < tol:
            break
    return rest2, P_partition_

def get_cr_one_path(partition, b_dict, r, T, opt_result):
    partition = partition[0]
    com_order = []
    combined = combinations_sort(partition, r, b_dict)
    total_b, total_r, total_commodity_order = zip(*combined)
    total_b = list(total_b)
    total_r = list(total_r)
    cr = get_CR(total_b, total_r, T, opt_result)
    com_order = list(zip(total_b, total_r, total_commodity_order))
    return cr, com_order

def p_list_extend(_p_list, L, pre_len):
    assert (L - pre_len - len(_p_list)) >= 0
    return [0] * pre_len + _p_list + [0] * (L - pre_len - len(_p_list))

def opt_result_extend(opt, L):
    assert L - len(opt) >= 0
    return opt + [opt[-1]] * (L - len(opt))

def P_list_and_get_ALG(b, r, min_c_max, T, opt_result, ret_alg=False):
    alg_l = []
    alg = 0
    c = min_c_max
    start_day = [1]
    p_list = []
    prob = []
    P_i = 0
    count = 0
    L = 0
    for i in range(1, T + 1):
        alg_l.append(alg)
        OPT_i = opt_result[i] - opt_result[i - 1]
        c_OPT_i = c * OPT_i
        flag = True
        try:
            while flag:
                flag = False
                if b[count] - r[count] != 0:
                    X_i = (c_OPT_i - (1 - P_i) * r[count] - sum(r[(count + 1):])) / (b[count] - r[count])
                    P_i = P_i + X_i
                else:
                    X_i = 1 - P_i
                    P_i = 1 + 1
                plus = X_i * b[count] + (1 - P_i - X_i) * r[count] + sum(r[count + 1:])
                alg += plus
                prob.append(X_i)
                L += 1
                if P_i >= 1:
                    if count == len(b) - 1:
                        P_i = P_i - X_i
                        prob.pop()
                        prob.append(1 - P_i)
                        alg = alg - plus + (1 - P_i) * b[count]
                        p_list.append(prob)
                        raise Over()
                    else:
                        if P_i > 1:
                            P_i = P_i - X_i
                            c_OPT_i = c_OPT_i - (1 - P_i) * b[count]
                            prob.pop()
                            prob.append(1 - P_i)
                            alg = alg - plus + (1 - P_i) * b[count]
                            p_list.append(prob)
                            prob = []
                            P_i = X_i = 0
                            count += 1
                            start_day.append(i)
                            flag = True
                        elif P_i == 1:
                            p_list.append(prob)
                            prob = []
                            P_i = X_i = 0
                            count += 1
                            start_day.append(i)
                            raise Equ()
                else:
                    if i == T:
                        p_list.append(prob)
                        raise Over
                    elif X_i < 0:
                        p_list.append(prob)
                        raise Over
                    else:
                        i = i + 1
        except Over:
            break
        except Equ:
            continue
    alg_l.append(alg)
    if ret_alg:
        return p_list, start_day, alg
    else:
        return p_list, start_day

def sample_evaluate(all_path, opt_result, end_T, ret_cr=True, ret_alg=False):
    path_choice = random.choices(list(range(all_path.path_num)), weights=all_path.path_prob)[0]
    select_path = all_path.getPath(path_choice)
    b, r, p_list = select_path.b, select_path.r, select_path.p_list
    buy_days = [0] * len(p_list)
    isbuy = [False] * len(p_list)
    for i in range(len(buy_days)):
        buy_days[i] = random.choices(list(range(len(p_list[i]))), weights=p_list[i])[0]
    cost = 0.0
    for t in range(end_T):
        if t in buy_days:
            idx = buy_days.index(t)
            cost += b[idx]
            isbuy[idx] = True
        for i in range(len(isbuy)):
            if not isbuy[i]:
                cost += r[i]
    if ret_cr:
        cr = cost / opt_result[end_T]
        return cr
    elif ret_alg:
        return cost
    else:
        return None