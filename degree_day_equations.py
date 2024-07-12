import math


def single_sine_no_cutoff(Tmax, Tmin, LTT):
    if Tmax <= LTT:
        return 0
    elif Tmin >= LTT:
        return ((Tmax + Tmin) / 2) - LTT
    elif Tmin < LTT and Tmax > LTT:
        W = (Tmax - Tmin) / 2
        X1 = math.asin((LTT - ((Tmax + Tmin) / 2)) / W)
        return (1 / math.pi) * (
            W * math.cos(X1) - ((LTT - ((Tmax + Tmin) / 2)) * ((math.pi / 2) - X1))
        )


# Example for other cutoffs would follow a similar structure as above with additional conditions


def single_sine_horizontal_cutoff(Tmax, Tmin, LTT, UTT):
    if Tmin >= UTT:
        return UTT - LTT
    elif Tmax <= LTT:
        return 0
    elif Tmin >= LTT and Tmax <= UTT:
        return ((Tmax + Tmin) / 2) - LTT
    elif Tmin < LTT and Tmax > LTT and Tmax <= UTT:
        W = (Tmax - Tmin) / 2
        X1 = math.asin((LTT - ((Tmax + Tmin) / 2)) / W)
        return (1 / math.pi) * (
            W * math.cos(X1) - ((LTT - ((Tmax + Tmin) / 2)) * ((math.pi / 2) - X1))
        )
    elif Tmin >= LTT and Tmax > UTT and Tmin < UTT:
        W = (Tmax - Tmin) / 2
        X2 = math.asin((UTT - ((Tmax + Tmin) / 2)) / W)
        return (1 / math.pi) * (W * (X2 - X1) + ((UTT - LTT) * ((math.pi / 2) - X2)))
    elif Tmin < LTT and Tmax > UTT:
        W = (Tmax - Tmin) / 2
        X1 = math.asin((LTT - ((Tmax + Tmin) / 2)) / W)
        X2 = math.asin((UTT - ((Tmax + Tmin) / 2)) / W)
        return (1 / math.pi) * (W * (X2 - X1) + ((UTT - LTT) * ((math.pi / 2) - X2)))


def single_sine_vertical_cutoff(Tmax, Tmin, LTT, UTT):
    if Tmin > UTT or Tmax < LTT:
        return 0
    elif Tmin > LTT and Tmax < UTT:
        return ((Tmax + Tmin) / 2) - LTT
    elif Tmin < LTT and Tmax > LTT and Tmax <= UTT:
        W = (Tmax - Tmin) / 2
        X1 = math.asin((LTT - ((Tmax + Tmin) / 2)) / W)
        return (1 / math.pi) * (
            W * math.cos(X1) - ((LTT - ((Tmax + Tmin) / 2)) * ((math.pi / 2) - X1))
        )
    elif Tmin > LTT and Tmax > UTT:
        W = (Tmax - Tmin) / 2
        X2 = math.asin((UTT - ((Tmax + Tmin) / 2)) / W)
        return (1 / math.pi) * (W * (X2 - X1) + ((UTT - LTT) * ((math.pi / 2) - X2)))
    elif Tmin < LTT and Tmax > UTT:
        W = (Tmax - Tmin) / 2
        X1 = math.asin((LTT - ((Tmax + Tmin) / 2)) / W)
        X2 = math.asin((UTT - ((Tmax + Tmin) / 2)) / W)
        return (1 / math.pi) * (W * (X2 - X1) + ((UTT - LTT) * ((math.pi / 2) - X2)))


def single_sine_intermediate_cutoff(Tmax, Tmin, LTT, UTT):
    if Tmin > UTT:
        return ((UTT + Tmin) / 2) - LTT
    elif Tmax < LTT:
        return 0
    elif Tmin > LTT and Tmax < UTT:
        return ((Tmax + Tmin) / 2) - LTT
    elif Tmin < LTT and Tmax > LTT and Tmax <= UTT:
        W = (Tmax - Tmin) / 2
        X1 = math.asin((LTT - ((Tmax + Tmin) / 2)) / W)
        return (1 / math.pi) * (
            W * math.cos(X1) - ((LTT - ((Tmax + Tmin) / 2)) * ((math.pi / 2) - X1))
        )
    elif Tmin > LTT and Tmax > UTT:
        W = (Tmax - Tmin) / 2
        X2 = math.asin((UTT - ((Tmax + Tmin) / 2)) / W)
        return (1 / math.pi) * (W * (X2 - X1) + ((UTT - LTT) * ((math.pi / 2) - X2)))
    elif Tmin < LTT and Tmax > UTT:
        W = (Tmax - Tmin) / 2
        X1 = math.asin((LTT - ((Tmax + Tmin) / 2)) / W)
        X2 = math.asin((UTT - ((Tmax + Tmin) / 2)) / W)
        return (1 / math.pi) * (W * (X2 - X1) + ((UTT - LTT) * ((math.pi / 2) - X2)))
