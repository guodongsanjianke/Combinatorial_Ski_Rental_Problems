from collections import defaultdict
import basic_func
from basic_func import shop
import numpy as np

epsilon = 1e-6


def csr(b, r, discount = None,fixed_partitions = None,b_dict = None,opt_lines=None):

    """
    :param b: The purchase price of each item
    :param r: The rent of each item
    :param discount: The increased discount rate for each additional item purchased
    :param b_dict: Purchase prices for different combinations of items
    :param fixed_partitions: The specified partitions of participating CR calculations
    :return: {'CR':CR,'all_shop':all_shop}
    """

    global epsilon,CR,prob

    n = len(b)
    product_list = list(range(1, n + 1))

    # If r is not specified
    # The rent for each item is set to 1
    if len(r) == 0:
        for j in range(n):
            r.append(1)

    if b_dict is None or opt_lines is None:
        two_partitions = list(basic_func.get_two_partitions(product_list))

    # If b_dict is not specified
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
        opt_result, T , _ = basic_func.get_opt_result(b_dict, r, two_partitions, product_list)
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
        CR = basic_func.get_CR(b,r,T,opt_result)
        return {
            'CR':CR
        }
        # prob = basic_func.P_list_and_get_ALG(b,r,CR,T,opt_result)
    else:
        # compute CR
        # and the probability of each partition
        CR, prob = basic_func.get_min_c_max(all_partitions, u_b, u_r, partitions_list, T, opt_result)

    # double check
    # as return result
    t_b,t_r = basic_func.renew_b_r(u_b, u_r, partitions_list, prob)
    CR = basic_func.get_CR(t_b, t_r, T, opt_result)
    p_list,start_day = basic_func.P_list_and_get_ALG(t_b, t_r, CR, T, opt_result)

    # This is just for alignment with test.py
    test_T = int(np.max(2*np.array(b)/np.array(r)))

    p_list_dict = defaultdict(lambda: defaultdict(str))

    # The probability of each combination is processed according to the result
    for i in range(len(p_list)):
        partitions_list[i].p_list = p_list[i]
        p_list_dict[partitions_list[i].p_id][partitions_list[i].split] = basic_func.p_list_extend(p_list[i], test_T, start_day[i] - 1)

    all_shop = []

    # Assemble the dict and shop constructs
    # The result is each partition case where the probability is positive,
    # and the probability of the combinations therein
    for index,partition in enumerate(all_partitions):
        if prob[index]==0:
            continue
        # Sort based on the ratio of prices to rent for product combinations
        combined_info = basic_func.combinations_sort(partition, r, b_dict)
        b_,r_,split = map(list,zip(*combined_info))
        p_list = []
        for s in split:
            p_list.append(p_list_dict[index][s])
        all_shop.append(shop(prob[index],p_list,b_,r_,split))

    return {
        'CR':CR,
        'all_shop':all_shop
    }


if __name__ == '__main__':

    Discount = 0.05
    B = [21.73, 40.1, 63.7]
    R = [6.21, 7.57, 2.78]
    # compute CR
    c = csr(B, R, Discount)['CR']
    print('results' + str(c))
