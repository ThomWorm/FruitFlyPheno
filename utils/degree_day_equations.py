import math
import numpy as np


def single_sine_horizontal_cutoff(T_min, T_max, LTT, UTT):
    if np.isnan(T_min) or np.isnan(T_max):
        return np.nan  # Return NaN if either input is NaN
    if T_min == -9999 or T_max == -9999:
        return np.nan
    alpha = (T_max - T_min) / 2
    theta = (T_max + T_min) / 2

    if T_min >= UTT:  # case 1
        return UTT - LTT
    elif T_max <= LTT:  # case 2
        return float(0)
    elif T_min >= LTT and T_max <= UTT:  # case 3

        return theta - LTT
    elif T_min < LTT and T_max > LTT and T_max <= UTT:  # case 4

        theta_1 = math.asin((LTT - theta) * (1 / alpha))

        return (1 / math.pi) * (
            (theta - LTT) * ((math.pi / 2) - theta_1) + (alpha * math.cos(theta_1))
        )

    elif T_min >= LTT and T_max > UTT and T_min < UTT:  # case 5
        theta_2 = math.asin((UTT - theta) * (1 / alpha))
        return (1 / math.pi) * (
            (theta - LTT) * (math.pi / 2 + theta_2)
            + (UTT - LTT) * (math.pi / 2 - theta_2)
            - alpha * math.cos(theta_2)
        )

    elif T_min < LTT and T_max > UTT:  # case 6
        theta_1 = math.asin((LTT - theta) * (1 / alpha))
        theta_2 = math.asin((UTT - theta) * (1 / alpha))

        return (1 / math.pi) * (
            (theta - LTT) * (theta_2 - theta_1)
            + alpha * (math.cos(theta_1) - math.cos(theta_2))
            + (UTT - LTT) * ((math.pi / 2) - theta_2)
        )


vsingle_sine_horizontal_cutoff = np.vectorize(single_sine_horizontal_cutoff)
