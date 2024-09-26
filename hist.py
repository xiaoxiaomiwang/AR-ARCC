import math
def calculate_cwc(picp, mean_width):
    cwc = mean_width * (1 + math.exp((-1.0) * (picp - 0.95)))
    return cwc

picp_value =0.970
mean_width_value = 30.683
cwc_value = calculate_cwc(picp_value, mean_width_value)

print("Coverage Width-based Criterion (CWC) = {:.4f}".format(cwc_value))

