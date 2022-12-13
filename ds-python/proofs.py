from cmath import e, pi, sqrt
from unicodedata import name
from scipy.special import gamma, betainc
import matplotlib.pyplot as plt 



C = 0.8
eps = 0.4
N = 3

# need to calculate the surface area, since as N grows the volume of a sphere --> 0

def volume_cap(r, a, n):
    gamma_value = gamma(n/2 + 1)
    beta_params_x = 1 - (a ** 2 / r ** 2)
    beta_params_a = (n + 1) / 2
    beta_params_b = 0.5
    beta_value = betainc(beta_params_a, beta_params_b, beta_params_x)
    if a >= 0:
        volume = 0.5 * (pow(pi, n/2) / gamma_value) * pow(r, n) * beta_value
    else:
        print("here")
        volume = 0.5 * (pow(pi, n/2) / gamma_value) - volume_cap(r, -a, n)
    return volume



def hypersphere_volume(d, r):
    denom = gamma(d/2 + 1)
    volume = (pow(pi, d/2) * pow(r, d)) / denom
    return volume
    
def volume_graph():
    radius = [1.0, 0.9, 0.95, 0.85, 0.8, 0.7]
    dimensions = [64, 128, 256]
    for d in dimensions:
        volume_list = list()
        for r in radius:
            volume_list.append(hypersphere_volume(d, r))
        print("dimension : {0}\n".format(d))
        for v, r1 in zip(radius, volume_list):
            print("{0}, {1}".format(r1, v))

def get_radius(v, d):
    t1 = pow((pi * d), (1 / 2*d))
    t2 = sqrt(d / (2 * pi * e))
    t3 = pow(v, 1/d)
    return t1 * t2 * t3

if __name__ == '__main__':
    # r = C
    # c1 = eps / 2
    # c2 = (pow(eps, 2) - 2 * pow(C, 2)) / (2*eps)
    # print(c2)
    # v1 = volume_cap(r, c1, N)
    # v2 = volume_cap(r, c2, N)
    # print(v1)
    # print(v2)
    # print("Volume of intersection for C={0}, eps={1}, N={2}: {3}".format(C, eps, N, v1 + v2))
    # volume_graph()
    # print(get_radius(1, 128))
    print(hypersphere_volume(128, 0.8))