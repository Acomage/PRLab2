from matplotlib import table
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import GPy
import os


def load_data() -> pd.DataFrame:
    data_path = os.path.join(
        os.path.dirname(__file__), "../../../homework3/data hw4 Ex2/toy_xy.csv"
    )
    data = pd.read_csv(data_path)
    return data


def GuassianProcess(
    x: np.ndarray,
    y: np.ndarray,
    variance: float = 1,
    lengthscale: float = 1,
    save: bool = False,
) -> float:
    k = GPy.kern.RBF(input_dim=1, variance=variance, lengthscale=lengthscale)
    m = GPy.models.GPRegression(x, y, k)
    if save:
        m.plot()
        img_path = os.path.join(
            os.path.dirname(__file__), f"../../img/var={variance}_len={lengthscale}.png"
        )
        plt.savefig(img_path)
    return np.exp(m._log_marginal_likelihood)


"""
This function is copy from the GPy tutorial:
https://nbviewer.org/github/gpschool/labs/blob/2021/2021/.answers/lab_1.ipynb
"""


def plot_gp(X, m, C, training_points=None):
    """Plotting utility to plot a GP fit with 95% confidence interval"""
    # Plot 95% confidence interval
    plt.fill_between(
        X[:, 0],
        m[:, 0] - 1.96 * np.sqrt(np.diag(C)),
        m[:, 0] + 1.96 * np.sqrt(np.diag(C)),
        alpha=0.5,
    )
    # Plot GP mean and initial training points
    plt.plot(X, m, "-")
    plt.legend(labels=["GP fit"])

    plt.xlabel("x"), plt.ylabel("f")

    # Plot training points if included
    if training_points is not None:
        X_, Y_ = training_points
        plt.plot(X_, Y_, "kx", mew=2)
        plt.legend(labels=["GP fit", "sample points"])


def task1():
    table_latex = "\\begin{table}[H]\n\\centering\n\\begin{tabular}{|l|"
    data = load_data()
    x = data["x"].values.reshape(-1, 1)
    y = data["y"].values.reshape(-1, 1)
    var_list = [0.5, 0.75, 1.0, 1.25, 1.5]
    len_list = [0.5, 0.75, 1.0, 1.25, 1.5]
    table_latex += (
        "c|" * len(len_list)
        + "}\n\\toprule\n\\diagbox{variance}{lengthscale} & "
        + " & ".join([str(i) for i in len_list])
        + " \\\\\n\\midrule\n"
    )
    for var in var_list:
        table_latex += f"{var} & "
        for length in len_list:
            table_latex += (
                f"{GuassianProcess(x, y, variance=var, lengthscale=length):.3e} & "
            )
        table_latex = table_latex[:-2] + "\\\\\n"
    table_latex += "\\bottomrule\n\\end{tabular}\n\\caption{Marginal likelihood and hyperparameters for RBF kernel}\n\\label{tab:ex2_1_1}\n\\end{table}"
    print(table_latex)


def task2():
    data = load_data()
    x = data["x"].values.reshape(-1, 1)
    y = data["y"].values.reshape(-1, 1)
    k = GPy.kern.RBF(1)
    m = GPy.models.GPRegression(x, y, k)
    m.optimize()
    print(np.exp(m._log_marginal_likelihood))
    print(m)


def task3():
    data = load_data()
    X = data["x"].values.reshape(-1, 1)
    Y = data["y"].values.reshape(-1, 1)
    k = GPy.kern.RBF(1)
    m = GPy.models.GPRegression(X, Y, k)
    m.optimize()
    k = m.kern
    Xnew = np.linspace(0.2, 20, 100)[:, None]
    Kxx = k.K(X, X) + 1 * np.eye(100)
    Ksx = k.K(Xnew, X)
    Kss = k.K(Xnew, Xnew)
    mean = Ksx @ np.linalg.inv(Kxx) @ Y
    Cov = Kss - Ksx @ np.linalg.inv(Kxx) @ Ksx.T
    plt.figure(figsize=(14, 8))
    plot_gp(Xnew, mean, Cov, training_points=(X, Y))
    plt.show()
    # plt.savefig(os.path.join(os.path.dirname(__file__), '../../img/task3.png'))


def task4():
    table_latex = "\\begin{table}[H]\n\\centering\n\\begin{tabular}{|l|c|c|}\n\\toprule\nX & posterior mean & posterior standard deviation \\\\\n\\midrule\n"
    data = load_data()
    X = data["x"].values.reshape(-1, 1)
    Y = data["y"].values.reshape(-1, 1)
    k = GPy.kern.RBF(1)
    m = GPy.models.GPRegression(X, Y, k)
    m.optimize()
    # print(m.predict_noiseless(np.array([[12], [22]])))
    for x in [12, 22]:
        mean, var = m.predict_noiseless(np.array([[x]]))
        table_latex += f"{x} & {mean[0][0]:.3e} & {np.sqrt(var[0][0]):.3e} \\\\\n"
    table_latex += "\\bottomrule\n\\end{tabular}\n\\caption{Predictive mean and variance at the test points}\n\\label{tab:ex2_1_4}\n\\end{table}"
    print(table_latex)


if __name__ == "__main__":
    # task1()
    # task2()
    task3()
    # task4()
