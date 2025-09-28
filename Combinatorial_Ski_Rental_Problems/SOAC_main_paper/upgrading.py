import basic_func

def csr_upgrading(b, r, discount=None,b_dict=None):
    '''

    :param b:  The purchase price of each item
    :param r:  The rent of each item
    :param discount: The increased discount rate for each additional item purchased
    :param b_dict: Purchase prices for different combinations of items
    :return:
    '''

    n = len(b)
    product_list = list(range(1, n + 1))

    # If r is not specified
    # The rent for each item is set to 1
    if len(r) == 0:
        for j in range(n):
            r.append(1)

    two_partitions = list(basic_func.get_two_partitions(product_list))
    if b_dict is None:
        assert discount is not None
        b_dict = {}
        for partition in two_partitions:
            b_dict[tuple(partition[0])] = sum([b[i - 1] for i in partition[0]])
            if len(partition[0]) > 1:
                b_dict[tuple(partition[0])] = b_dict[tuple(partition[0])] * (1 - discount * (len(partition[0]) - 1))

    assert b_dict is not None

    opt_result, T, opt_group = basic_func.get_opt_result(b_dict, r, two_partitions, product_list)

    u_b, u_r, partitions_list = basic_func.transfer_for_upgrading(opt_group, b_dict, r)

    CR = basic_func.get_CR(u_b, u_r, T, opt_result)

    return CR


if __name__ == '__main__':

    Discount = 0.05
    B = [55.0, 45.0]
    R = [2, 4]
    # compute CR
    c = csr_upgrading(B, R, Discount)
    print('Result:' + str(c))
