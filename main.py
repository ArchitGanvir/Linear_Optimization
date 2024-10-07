# Archit Ganvir (CS21BTECH11005)
# Prasham Walvekar (CS21BTECH11047)


import numpy as np
import scipy as sp


def inputHandler(input_file):
    input = np.loadtxt(input_file, delimiter=",")

    n = input.shape[1] - 1

    c = input[0, :n]

    A = input[1:, :n]

    b = input[1:, n]

    return  n, c, A, b

def func2(n, c, A, b):
    zed = np.min(b)

    n = n + 1

    z = np.zeros(n)
    z[-1] = zed

    c = np.append(c, 0)

    A = np.append(A, np.ones((A.shape[0], 1)), axis=1)
    R_n = np.zeros(n)
    R_n[-1] = -1
    A = np.append(A, [R_n], axis=0)

    b = b + zed
    b = np.append(b, -zed)

    return n, z, c, A, b

def steps_2_and_3(n, z, c, A, b, b_calculated):
    # Step 2

    tight_rows = np.isclose(b_calculated, b)

    A1 = A[tight_rows]

    while (isVertex(A1, n) == False):
        untight_rows = np.logical_not(tight_rows)

        A2 = A[untight_rows]

        if (np.size(A1) == 0):
            while True:
                u = np.random.rand(n)

                values_for_denominator, rows_for_non_zero_denominator, any_rows_for_non_zero_denominator = func1(A2, u)

                if (any_rows_for_non_zero_denominator == True):
                    break
        else:
            S = sp.linalg.null_space(A1)
            for i in range(S.shape[1]):
                u = S[:, i]

                values_for_denominator, rows_for_non_zero_denominator, any_rows_for_non_zero_denominator = func1(A2, u)

                if (any_rows_for_non_zero_denominator == True):
                    break

        b2 = b[untight_rows]
        b_calculated2 = b_calculated[untight_rows]

        values_for_alpha = (b2[rows_for_non_zero_denominator] - b_calculated2[rows_for_non_zero_denominator]) / values_for_denominator[rows_for_non_zero_denominator]

        absolute_values_for_alpha = np.abs(values_for_alpha)

        index_of_minimum_absolute_value_for_alpha = np.argmin(absolute_values_for_alpha)

        alpha = values_for_alpha[index_of_minimum_absolute_value_for_alpha]

        z = z + alpha * u

        b_calculated = np.dot(A, z)
        tight_rows = np.isclose(b_calculated, b)

        A1 = A[tight_rows]

        print(z, np.dot(c, z))

    # Step 3

    if (degeneracyTest(tight_rows, n) == True):
        handle_degeneracy(n, z, c, A, b, b_calculated)

    A1_inverse = np.linalg.inv(A1)

    alpha = np.dot(c, A1_inverse)

    indices_for_negative_alpha = np.where(np.logical_and(alpha < 0, np.logical_not(np.isclose(alpha, 0))))

    while(isOptimumVertex(indices_for_negative_alpha) == False):
        v = -A1_inverse[:, indices_for_negative_alpha[0][0]]

        untight_rows = np.logical_not(tight_rows)

        A2 = A[untight_rows]

        values_for_denominator = np.dot(A2, v)

        rows_with_positive_denominator = np.logical_and(values_for_denominator > 0, np.logical_not(np.isclose(values_for_denominator, 0)))

        if (np.any(rows_with_positive_denominator) == False):
            print("The polytope/system is unbounded and the objective function is also unbounded")

            exit()

        b2 = b[untight_rows]
        b_calculated2 = b_calculated[untight_rows]

        values_for_beta = (b2[rows_with_positive_denominator] - b_calculated2[rows_with_positive_denominator]) / values_for_denominator[rows_with_positive_denominator]

        index_of_minimum_value_for_beta = np.argmin(values_for_beta)

        beta = values_for_beta[index_of_minimum_value_for_beta]

        z = z + beta * v

        b_calculated = np.dot(A, z)
        tight_rows = np.isclose(b_calculated, b)
        A1 = A[tight_rows]

        if (degeneracyTest(tight_rows, n) == True):
            print(z, np.dot(c, z))
            handle_degeneracy(n, z, c, A, b, b_calculated)

        A1_inverse = np.linalg.inv(A1)
        alpha = np.dot(c, A1_inverse)
        indices_for_negative_alpha = np.where(np.logical_and(alpha < 0, np.logical_not(np.isclose(alpha, 0))))

        print(z, np.dot(c, z))

    return A1_inverse, tight_rows, z

def isVertex(A1, n):
    if (np.size(A1) == 0):
        return False
    
    return np.linalg.matrix_rank(A1) == n

def func1(A2, u):
    values_for_denominator = np.dot(A2, u)

    rows_with_zero_denominator = np.isclose(values_for_denominator, 0)

    rows_for_non_zero_denominator = np.logical_not(rows_with_zero_denominator)

    any_rows_for_non_zero_denominator = np.any(rows_for_non_zero_denominator)

    return values_for_denominator, rows_for_non_zero_denominator, any_rows_for_non_zero_denominator

def isOptimumVertex(indices_for_negative_alpha):
    if (np.size(indices_for_negative_alpha) == 0):
        return True
    
    return False

def degeneracyTest(tight_rows, n):
    if (np.count_nonzero(tight_rows) > n):
        return True
    
    return False

def minimumAbsoluteNonZeroValueIn(a):
    flattened_a = a.flatten()

    non_zero_values = flattened_a[np.logical_not(np.isclose(flattened_a, 0))]

    absolute_non_zero_values = np.abs(non_zero_values)

    minimum_absolute_non_zero_value = np.min(absolute_non_zero_values)

    return minimum_absolute_non_zero_value

def newb(A, b):
    minimum_absolute_non_zero_value_in_A = minimumAbsoluteNonZeroValueIn(A)
    minimum_absolute_non_zero_value_in_b = minimumAbsoluteNonZeroValueIn(b)

    minimum_absolute_non_zero_value_in_A_and_b = min(minimum_absolute_non_zero_value_in_A, minimum_absolute_non_zero_value_in_b)

    epsilon = minimum_absolute_non_zero_value_in_A_and_b / 2

    b = b.copy()

    for i in range(b.size):
        b[i] = b[i] + epsilon ** (i + 1)

    return b

def handle_degeneracy(n, z, c, A, b, b_calculated):
    b_new = newb(A, b)

    A1_inverse, tight_rows, z = steps_2_and_3(n, z, c, A, b_new, b_calculated)

    b1 = b[tight_rows]

    z = np.dot(A1_inverse, b1)

    print(z, np.dot(c, z))

    z_original_optimum = z[:-1]

    print(z_original_optimum, np.dot(c, z))

    exit()


input_file = open("./Assignment 4/b.csv", "r")

n, c, A, b = inputHandler(input_file)

n, z, c, A, b = func2(n, c, A, b)

# Step 1

b_calculated = np.dot(A, z)

print(z, np.dot(c, z))

# Steps 2 and 3

A1_inverse, tight_rows, z = steps_2_and_3(n, z, c, A, b, b_calculated)

z_original_optimum = z[:-1]

print(z_original_optimum, np.dot(c, z))

input_file.close()
