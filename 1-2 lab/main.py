from typing import *
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats as sci
import math

class Distribution:
    def __init__(self, name: str,
                 generate: Callable[[int], np.ndarray],
                 prob_density: Callable[[float], float],
                 discrete = False) -> None:
        self.name = name
        self.generate = generate
        self.prob_density = prob_density
        self.discrete = discrete

def draw_hist(distribution: Distribution, size: int, folder: str = ".", ext: str = "pdf"):
    data = distribution.generate(size)

    bar_count = round(1 + 3.322*np.log10(size))
    bar_width = (data.max() - data.min()) / bar_count
    
    x = np.linspace(data.min(), data.max(), 1000) if not distribution.discrete else np.arange(data.min(), data.max())
    
    plt.clf()
    plt.hist(data, density=True, bins=bar_count, label="Generated data", edgecolor='black', linewidth=1.0)
    plt.plot(x, [distribution.prob_density(x) for x in x], label="Density")
    plt.title(f"{distribution.name.capitalize()}, $n = {size}$")
    plt.xlabel(f"{distribution.name.capitalize()} numbers")
    plt.ylabel("Density")
    plt.legend()
    #plt.show()

    if not os.path.isdir(folder):
        os.makedirs(folder)

    plt.savefig(f"{folder}/hist_{distribution.name.casefold().replace(' ', '_')}_{size}.{ext}")

def Mean(sorted_data: np.ndarray):
    return np.mean(sorted_data)

def Variance(sorted_data: np.ndarray):
    square = np.array([x ** 2 for x in sorted_data])
    return np.mean(square) - np.mean(sorted_data) ** 2

def mediana(sorted_data: np.ndarray):
    return np.median(sorted_data)

def sum_of_extr(sorted_data: np.ndarray):
    return (sorted_data[0] + sorted_data[-1]) / 2

def quartile_semisum(sorted_data: np.ndarray):
    return (np.quantile(sorted_data, 0.25) + np.quantile(sorted_data, 0.75)) / 2

def truncated_mean(sorted_data: np.ndarray):
    n = len(sorted_data)
    r = int(n / 4)
    summ = 0
    for i in range(r + 1, n - r + 1):
        summ += sorted_data[i] / (n - 2 * r)
    return summ

def generate_data(distribution: Distribution, characteristic: Callable[[np.ndarray], float], size: int, repeats: int = 1000):
    res = np.array([])
    for _ in range(repeats):
        data = distribution.generate(size)
        sorted_data = np.sort(data)
        res = np.append(res, characteristic(sorted_data))
    return res


def get_characteristics(distribution: Distribution, sizes: List[int], repeats: int = 1000, folder: str = ".", ext: str = "tex"):
    chars = [Mean, mediana, sum_of_extr, quartile_semisum, truncated_mean]
    chars_head = ["$\\overline{x}$", "$med\\ x$", "$z_R$", "$z_Q$", "$z_{tr}$"]

    upper_chars = [Mean, Variance]
    upper_chars_head = ["$E(z)$", "$D(z)$"]

    filename = f"{folder}/chars_{distribution.name.casefold().replace(' ', '_')}.{ext}"

    if not os.path.isdir(folder):
        os.makedirs(folder)

    with open(filename, 'w', encoding='utf-8') as file:
        file.write("\\begin{tabular}{| ")
        for _ in range(len(chars) + 1):
            file.write("c |")
        file.write("} \\hline \n")

        for size in sizes:
            file.write(f"{distribution.name.capitalize()}, n = {size}")
            for _ in chars:
                file.write(" &")
            file.write("\\\\ \\hline \n")

            file.write(f" & {' & '.join(chars_head)} \\\\ \\hline \n")
            mean = [Mean(generate_data(distribution, char, size, repeats)) for char in chars]
            file.write(f"$E(z)$ & {' & '.join([np.round(x, 4).astype(str) for x in mean])} \\\\ \\hline \n")
            variance = [Variance(generate_data(distribution, char, size, repeats)) for char in chars]
            file.write(f"$D(z)$ & {' & '.join([np.round(x, 4).astype(str) for x in variance])} \\\\ \\hline \n")


            for _ in chars:
                file.write("& ")
            file.write("\\\\ \\hline \n")

        file.write("\\end{tabular}")

distributions = [
    Distribution(
        name="normal",
        generate=lambda size: sci.norm.rvs(loc=0, scale=1, size=size),
        prob_density=lambda x: sci.norm.pdf(x, loc=0, scale=1),
    ),
    Distribution(
        name="cauchy",
        generate=lambda size: sci.cauchy.rvs(loc=0, scale=1, size=size),
        prob_density=lambda x: sci.cauchy.pdf(x, loc=0, scale=1),
    ),
    Distribution(
        name="student",
        generate=lambda size: sci.t.rvs(3, loc=0, scale=1 / np.sqrt(2), size=size),
        prob_density=lambda x: sci.t.pdf(x, 3, loc=0, scale=1 / np.sqrt(2)),
    ),
    Distribution(
        name="poisson",
        generate=lambda size: sci.poisson.rvs(10, loc=0, size=size),
        prob_density=lambda x: sci.poisson.pmf(x, 10, loc=0),
        discrete=True
    ),
    Distribution(
        name="uniform",
        generate=lambda size: sci.uniform.rvs(loc=-np.sqrt(3), scale=2 * np.sqrt(3), size=size),
        prob_density=lambda x: sci.uniform.pdf(x, loc=-np.sqrt(3), scale=2 * np.sqrt(3)),
    )
]

def first_lab():
    for size in [10, 50, 1000]:
        for distribution in distributions:
            draw_hist(distribution, size, folder="./hists")

def second_lab():
    for distribution in distributions:
        get_characteristics(distribution, [10, 100, 1000], folder="./chars")

if __name__ == "__main__":
    np.random.seed(13371488)
    first_lab()
    second_lab()
    
