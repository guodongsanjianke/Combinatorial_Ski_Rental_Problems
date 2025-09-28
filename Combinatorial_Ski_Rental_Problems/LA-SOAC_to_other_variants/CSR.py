from collections import defaultdict
import basic_func
from basic_func import path, All_Path, sample_evaluate
from tqdm import trange
import numpy as np
import random
import matplotlib.pyplot as plt

epsilon = 1e-6


def csr(b, r, discount=None, fixed_partitions=None, b_dict=None, opt_lines=None, eval_T=None):
    """
    :param b: The purchase price of each item
    :param r: The rent of each item
    :param discount: The increased discount rate for each additional item purchased
    :param b_dict: Purchase prices for different combinations of items
    :param fixed_partitions: The specified partitions of participating CR calculations
    :return: {'CR':CR,'all_path':all_path}
    """

    global epsilon, CR, prob, two_partitions

    n = len(b)
    product_list = list(range(1, n + 1))

    # If r is not specified
    # The rent for each item is set to 1
    if len(r) == 0:
        for j in range(n):
            r.append(1)

    if b_dict is None or opt_lines is None:
        two_partitions = list(basic_func.get_two_partitions(product_list))

    # If b_dict is not specified\
    # The price of the commodity combination is deduced from the b, r and discount
    if b_dict is None:
        assert discount is not None
        b_dict = {}
        for partition in two_partitions:
            b_dict[tuple(partition[0])] = sum([b[i - 1] for i in partition[0]])
            if len(partition[0]) > 1:
                b_dict[tuple(partition[0])] = b_dict[tuple(partition[0])] * (1 - discount * (len(partition[0]) - 1))

    assert b_dict is not None

    # get the opt result
    if opt_lines is None:
        opt_result, T, _ = basic_func.get_opt_result(b_dict, r, two_partitions, product_list)
    else:
        opt_result, T = basic_func.get_opt_result1(opt_lines)

    assert opt_result is not None and T is not None

    # If no partition cases participating in the calculation are specified,
    # all partition cases are used for the calculation
    if fixed_partitions is not None:
        all_partitions = fixed_partitions
    else:
        all_partitions = basic_func.get_all_partitions(product_list[::-1])

    assert len(all_partitions) > 0

    # Construct the price of each combination based on probabilities
    u_b, u_r, partitions_list = basic_func.transfer(all_partitions, b_dict, r)

    if n == 1:
        CR = basic_func.get_CR(b, r, T, opt_result)
        return {
            'CR': CR
        }
        # prob = basic_func.P_list_and_get_ALG(b,r,CR,T,opt_result)
    else:
        # compute CR
        # and the probability of each partition
        CR, prob = basic_func.get_min_c_max(all_partitions, u_b, u_r, partitions_list, T, opt_result)

    # double check
    # as return result
    t_b, t_r = basic_func.renew_b_r(u_b, u_r, partitions_list, prob)
    CR = basic_func.get_CR(t_b, t_r, T, opt_result)
    p_list, start_day = basic_func.P_list_and_get_ALG(t_b, t_r, CR, T, opt_result)

    if not eval_T:
        eval_T = int(np.max(2 * np.array(b) / np.array(r)))

    assert eval_T is not None

    p_list_dict = defaultdict(lambda: defaultdict(str))

    # The probability of each combination is processed according to the result
    for i in range(len(p_list)):
        partitions_list[i].p_list = p_list[i]
        p_list_dict[partitions_list[i].p_id][partitions_list[i].split] = basic_func.p_list_extend(p_list[i], eval_T,
                                                                                                  start_day[i] - 1)

    opt_result_extend = basic_func.opt_result_extend(opt_result, eval_T)

    all_path = All_Path(n=len(all_partitions))

    # Assemble the dict and shop constructs
    # The result is each partition case where the probability is positive,
    # and the probability of the combinations therein
    for index, partition in enumerate(all_partitions):


        # Sort based on the ratio of prices to rent for product combinations
        combined_info = basic_func.combinations_sort(partition, r, b_dict)
        b_, r_, split = map(list, zip(*combined_info))
        p_list = []
        for s in split:
            p_list.append(p_list_dict[index][s])
        all_path.addPath(path(prob[index], p_list, b_, r_, split))

    return {
        'CR': CR,
        'all_path': all_path,
        'opt_result': opt_result_extend,
        'eval_T': eval_T,
    }


def evaluate(all_path, opt_result, eval_T,num_evaluate=10000):
    eval_cr = []
    for end_T in trange(1, eval_T):
        temp = []
        for _ in range(num_evaluate):
            cr = sample_evaluate(all_path, opt_result, end_T)
            temp.append(cr)
        eval_cr.append(np.mean(temp))

    return eval_cr


def plot_scatter_with_hline(data_list, hline_value, plot_color='blue', line_color='red'):
    """
    Draws a scatter plot and adds a horizontal dashed line.

    Parameters:
    - data_list: list[float], the data to be plotted
    - hline_value: float, position of the horizontal dashed line
    - dot_color: str, scatter color
    - line_color: str, color of the horizontal line
    """
    x = list(range(len(data_list)))
    y = data_list

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, color=plot_color, label='Data Points')
    plt.axhline(y=hline_value, color=line_color, linestyle='--', label=f'y = {round(hline_value, 4)}')
    # plt.axhline(y=np.mean(y), color='green', linestyle='--', label=f'y = {np.mean(y)}')

    plt.title('Real Evaluation')
    plt.xlabel('End Time')
    plt.ylabel('CR')
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    plt.savefig('eval.png', dpi=500)


if __name__ == '__main__':
    Discount = 0.05
    B = [21.73, 40.1, 63.7]
    R = [6.21, 7.57, 2.78]
    # eval_T =
    # compute CR
    alg_result = csr(B, R, Discount)
    CR, all_path, opt_result, eval_T = alg_result['CR'], alg_result['all_path'], alg_result['opt_result'], alg_result['eval_T']
    print(f"cr: {CR}")

    # evaluate
    eval_cr = evaluate(all_path, opt_result, eval_T)

    # draw

    plot_scatter_with_hline(eval_cr, CR)
