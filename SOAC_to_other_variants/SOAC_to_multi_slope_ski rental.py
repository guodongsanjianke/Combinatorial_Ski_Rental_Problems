import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def opt(b, r):
    opt_result = []
    for i in range(10 * K):
        t = i / K
        cost = [b[j] + r[j] * t for j in range(3)]
        opt_cost = min(cost)
        opt_result.append(opt_cost)
    return opt_result

class Equ(Exception):
    pass

class Over(Exception):
    pass

def get_CR(b, r, T, opt_result):
    c_max = math.exp(1)
    c_min = 1
    c = 2
    epsilon = 1e-6

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
                        P_i = 1

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

def P_list_and_get_ALG(b, r, min_c_max, T, opt_result):
    alg = 0
    c = min_c_max

    # The days on which each set of products begins to be purchased.
    start_day = [1]

    p_list = []
    prob = []

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
                    X_i = (c_OPT_i - (1 - P_i) * r[count] - sum(r[(count + 1):])) / (
                            b[count] - r[count])
                    P_i = P_i + X_i
                else:
                    X_i = 1 - P_i
                    P_i = 1

                prob.append(X_i)
                plus = X_i * b[count] + (1 - P_i - X_i) * r[count] + sum(r[count + 1:])
                alg += plus

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
                            P_i = 0
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

    return p_list, start_day, alg


def Draw(y1, y2, y3):
    mpl.rcParams['font.family'] = 'Times New Roman'

    x = [i / len(y1) for i in range(len(y1))]

    color = ['#195aff','#ffbe19','#c0ef17']

    plt.plot(x, y1,color =color[0])
    plt.plot(x, y2,linestyle = (0,(18,15)),color = color[1])
    plt.plot(x, y3,linestyle = (0,(10,5)),color = color[2])

    plt.title(f'Fig1 (Multi-slope)',size=16)
    plt.xlabel('Times',size=14)
    plt.ylabel('Probability',size=14)

    # plt.legend()

    plt.savefig('Fig1.png',dpi=600)

    plt.show()


# Init

# The purchase and rent prices of the first two items
b0 = 0
r0 = 2
b1 = 0.5
r1 = 0.5

# The purchase and rent price of the third item
b2 = [0.9, 0.7, 0.55]
r2 = [0.1, 0.3, 0.45]

K = 1000


for i in range(0,3):

    b = np.array([b0, b1, b2[i]])
    r = np.array([r0, r1, r2[i]])
    opt_list = opt(b, r)

    b_ = np.array([b1 - b0, b2[i] - b1])
    r_ = np.array([r0 - r1, r1 - r2[i], r2[i]])
    r_ = r_ / K
    T = 9*K

    c = get_CR(b_, r_, T, opt_list)
    print(c)
    prob_list, _, _ = P_list_and_get_ALG(b_, r_, c, T, opt_list)

    # Prepare the data for drawing

    y2 = []
    for i in range(len(prob_list[0])):
        if i == 0:
            y2.append(prob_list[0][i])
        else:
            y2.append(y2[-1] + prob_list[0][i])

    y3 = []
    for i in range(len(prob_list[1])):
        if i == 0:
            y3.append(prob_list[1][i])
        else:
            y3.append(y3[-1] + prob_list[1][i])

    y3 = y3[:K + 1 - len(y2)]

    l1 = len(y2)
    l2 = len(y3)

    y2 = np.array(y2)
    y1 = 1 - y2
    y3 = np.array(y3)
    y2 = np.concatenate((y2, 1 - y3))

    temp = np.zeros(l2)
    y1 = np.concatenate((y1, temp))

    temp = np.zeros(l1)
    y3 = np.concatenate((temp, y3))


    y1 = y1[0:K+1]
    y2 = y2[0:K+1]
    y3 = y3[0:K+1]

    Draw(y1, y2, y3)
