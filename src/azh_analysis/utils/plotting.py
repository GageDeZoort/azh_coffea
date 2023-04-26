from __future__ import annotations

import warnings

import hist
import mplhep as hep
import numpy as np
from cycler import cycler
from hist import Hist
from hist.intervals import ratio_uncertainty
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")


def plot_mc(
    mc,
    var,
    var_label,
    cat_label,
):
    hep.style.use(["CMS", "fira", "firamath"])
    colors = {
        "DY": "#0A9396",
        "SM-H(125)": "#E9D8A6",
        "ZZ": "#94D2BD",
        "WZ": "#9b2226",
        "tt": "#EE9B00",
        "VVV": "#bb3e03",
    }

    # grab the correct MC histogram
    group_hists = {}
    for group in colors.keys():
        try:
            group_hists[group] = mc[group, :]
        except Exception:
            dummy_axis = mc["ZZ", :].axes[var]
            group_hists[group] = Hist(dummy_axis)
            continue
    stack = hist.Stack.from_dict(group_hists)

    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(12, 12),
        dpi=200,
    )
    ax.set_prop_cycle(cycler(color=list(colors.values())))
    stack.plot(ax=ax, stack=True, histtype="fill", edgecolor=(0, 0, 0, 0.3))

    ax.set_ylabel("Counts")
    ax.set_xlabel(var_label)
    ax.legend()
    ax.legend(loc="best", prop={"size": 16}, frameon=True)
    ax.get_legend().set_title(f"{cat_label}")

    hep.cms.label("Preliminary", data=True, lumi=59.7, year=2018, ax=ax)
    plt.show()


def plot_data_vs_mc(
    data,
    mc,
    var,
    cat_label,
    var_label,
    logscale=False,
):
    hep.style.use(["CMS", "fira", "firamath"])
    colors = {
        "Reducible": "#005F73",
        "DY": "#0A9396",
        "SM-H(125)": "#E9D8A6",
        "ZZ": "#94D2BD",
        "WZ": "#9b2226",
        "tt": "#EE9B00",
        "VVV": "#bb3e03",
    }

    # grab the correct MC histogram
    group_hists = {"Reducible": data["reducible", :]}
    for group in colors.keys():
        if group == "Reducible":
            continue
        try:  # assuming the group has been populated
            group_hists[group] = mc[group, :]
        except Exception:  # use a dummy axis to display empty fields
            print(data["reducible", :].axes)
            dummy_axis = data["reducible", :].axes[var]
            group_hists[group] = Hist(dummy_axis)
            # print(f"{group} not in file")
            continue

    stack = hist.Stack.from_dict(group_hists)

    fig, (ax, rax) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 16),
        dpi=200,
        gridspec_kw={"height_ratios": (4, 1)},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.07)
    ax.set_prop_cycle(cycler(color=list(colors.values())))
    stack.plot(ax=ax, stack=True, histtype="fill", edgecolor=(0, 0, 0, 0.3))
    data["data", :].plot1d(ax=ax, histtype="errorbar", color="k", label="Data")

    mc_vals = sum(list(group_hists.values())).values()
    data_vals = data["data", :].values()
    bins = data["data", :].axes[0].centers
    rax.errorbar(
        x=bins,
        y=data_vals / mc_vals,
        yerr=ratio_uncertainty(data_vals, mc_vals, "poisson"),
        color="k",
        linestyle="none",
        marker="o",
        elinewidth=1,
    )

    if logscale:
        ax.set_yscale("log")
    ax.set_ylabel("Counts")
    rax.set_xlabel(var_label)
    ax.set_xlabel("")
    ax.legend()
    rax.set_ylim([0, 2])
    ax.legend(loc="best", prop={"size": 16}, frameon=True)
    ax.get_legend().set_title(f"{cat_label}")

    hep.cms.label("Preliminary", data=True, lumi=59.7, year=2018, ax=ax)
    plt.show()


def plot_var_fake_val(
    data,
    reducible,
    var,
    cat_label,
    var_label,
    xlim=None,
    xscale=None,
):
    hep.style.use(["CMS", "fira", "firamath"])
    colors = {
        "Reducible": "#005F73",
    }

    stack = hist.Stack.from_dict({"Reducible": reducible})

    fig, (ax, rax) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 16),
        dpi=200,
        gridspec_kw={"height_ratios": (4, 1)},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.07)
    ax.set_prop_cycle(cycler(color=list(colors.values())))
    stack.plot(ax=ax, stack=True, histtype="fill", edgecolor=(0, 0, 0, 0.3))
    data.plot1d(ax=ax, histtype="errorbar", color="k", label="Data")

    red_vals = reducible.values()
    data_vals = data.values()
    bins = data.axes[0].centers
    rax.errorbar(
        x=bins,
        y=data_vals / red_vals,
        yerr=ratio_uncertainty(data_vals, red_vals, "poisson"),
        color="k",
        linestyle="none",
        marker="o",
        elinewidth=1,
    )

    if xlim is not None:
        ax.set_xlim(xlim)
    if xscale is not None:
        ax.set_xscale(xscale)
    ax.set_ylabel("Counts")
    rax.set_xlabel(var_label)
    ax.set_xlabel("")
    ax.legend()
    rax.set_ylim([0, 2])
    ax.legend(loc="best", prop={"size": 16}, frameon=True)
    ax.get_legend().set_title(f"{cat_label}")

    hep.cms.label("Preliminary", data=True, lumi=59.7, year=2018, ax=ax)
    plt.show()


def get_ratios(denom, num, combine_bins=True):
    num_sum = num.values()
    denom_sum = denom.values()
    edges = denom.axes[-1].edges
    centers = denom.axes[-1].centers
    if combine_bins:
        bin_50n = num_sum[4] + num_sum[5]
        bin_70n = num_sum[6] + num_sum[7]
        bin_90n = num_sum[8] + num_sum[9]
        bin_50d = denom_sum[4] + denom_sum[5]
        bin_70d = denom_sum[6] + denom_sum[7]
        bin_90d = denom_sum[8] + denom_sum[9]
        num_sum = np.array(list(num_sum[:4]) + [bin_50n, bin_70n, bin_90n])
        denom_sum = np.array(list(denom_sum[:4]) + [bin_50d, bin_70d, bin_90d])
        edges = np.array(list(edges[:5]) + [60, 80, 100])
        centers = np.array(list(centers[:4]) + [50, 70, 90])

    ratios = np.nan_to_num(num_sum / denom_sum)
    mask = (num_sum > 0) & (denom_sum > 0) & (num_sum / denom_sum <= 1)
    centers, ratios, num_sum, denom_sum = (
        centers[mask],
        ratios[mask],
        num_sum[mask],
        denom_sum[mask],
    )
    uncerts = ratio_uncertainty(num_sum, denom_sum, "efficiency")
    x_err = ((edges[1:] - edges[:-1]) / 2)[mask]
    return edges, centers, ratios, uncerts, x_err


def fit_polynomial(ax, centers, ratios, yerr):
    mask = ratios > 0
    # std = yerr[1][mask]
    c, cov = np.polyfit(
        centers[mask],
        ratios[mask],
        2,
        cov=True,
        rcond=None,
        # w=1/std,
    )
    x = np.linspace(0, 120, 1000)
    ax.plot(
        x, c[2] + c[1] * x + c[0] * x**2, color="red", ls="--", label="Quadratic Fit"
    )
    return c[2] + c[1] * centers + c[0] * centers**2


def plot_fake_rate_measurements(
    denom,
    num,
    label,
    xlim,
    ylim,
    combine_bins=True,
):
    hep.style.use(["CMS", "fira", "firamath"])
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10), dpi=200)
    edges, centers, ratio, uncerts, bin_errs = get_ratios(denom, num, combine_bins)
    ax.errorbar(
        x=centers,
        y=ratio,
        yerr=uncerts,
        xerr=bin_errs,
        color="k",
        linestyle="none",
        marker="o",
        elinewidth=1.25,
        capsize=2,
        label="Data - Prompt MC",
    )
    ratio = fit_polynomial(ax, centers, ratio, uncerts)
    ax.set_xlim(xlim)
    ax.set_xlabel(r"$p_T$ [GeV]")
    ax.set_ylabel("Fake Rate")
    ax.legend(loc="best", prop={"size": 16}, frameon=True)
    ax.set_ylim(ylim)
    ax.get_legend().set_title(f"{label}", prop={"size": 16})
    hep.cms.label("Preliminary", data=True, lumi=59.7, year=2018, ax=ax)
    plt.show()
    return edges, centers, ratio, uncerts, bin_errs


def plot_fake_rates_data(
    denom,
    num,
    label,
    xlim,
    ylim,
    combine_bins=True,
):
    hep.style.use(["CMS", "fira", "firamath"])
    colors = ["#94d2bd", "#0a9396"]

    fig, (ax, rax) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 12),
        dpi=200,
        gridspec_kw={"height_ratios": (4, 1)},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.07)
    ax.set_prop_cycle(cycler(color=colors))

    denom.plot(
        ax=ax,
        stack=False,
        histtype="fill",
        edgecolor=(0, 0, 0, 0.3),
        label="Denominator",
    )

    num.plot(
        ax=ax, stack=False, histtype="fill", edgecolor=(0, 0, 0, 0.3), label="Numerator"
    )

    edges, centers, ratio, uncerts, bin_errs = get_ratios(denom, num, combine_bins)
    rax.errorbar(
        x=centers,
        y=ratio,
        yerr=uncerts,
        xerr=bin_errs,
        color="k",
        linestyle="none",
        marker="o",
        elinewidth=1.25,
        capsize=2,
    )

    ax.set_xlim(xlim)
    ax.set_yscale("log")
    ax.set_xlabel(r"$p_T$")
    ax.set_ylabel("Counts")
    rax.set_xlabel(ax.get_xlabel() + " [GeV]")
    ax.set_xlabel("")
    ax.legend(loc="best", prop={"size": 16}, frameon=True)
    rax.set_ylim(ylim)
    ax.get_legend().set_title(f"{label}", prop={"size": 16})
    hep.cms.label("Preliminary", data=True, lumi=59.7, year=2018, ax=ax)
    plt.show()


def plot_m4l_systematic(
    nom, up, down, syst, cat_label, mass_label, logscale=False, outfile=None
):
    fig, axs = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(30, 16),
        dpi=200,
        gridspec_kw={"height_ratios": (4, 1, 1)},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.07)
    colors = {"up": "#45ADA8", "down": "#FF4E50", "nom": "#2A363B"}
    for i, mass_type in enumerate(["raw", "corr", "cons"]):
        nom_s = nom[mass_type, :]
        shift_up = up[mass_type, :]
        shift_down = down[mass_type, :]

        n, u, d = nom_s.values(), shift_up.values(), shift_down.values()
        bin_centers = nom_s.axes[0].centers
        # un_ratio, un_err = np.nan_to_num(n / u), ratio_uncertainty(n, u, "poisson")
        # dn_ratio, dn_err = np.nan_to_num(n / d), ratio_uncertainty(n, d, "poisson")

        shift_up.plot1d(
            ax=axs[0, i], label=f"{syst} up", histtype="step", color=colors["up"]
        )
        shift_down.plot1d(
            ax=axs[0, i], label=f"{syst} down", histtype="step", color=colors["down"]
        )
        nom_s.plot1d(
            ax=axs[0, i], label=f"{syst} nom", histtype="step", color=colors["nom"]
        )

        if logscale:
            axs[0, i].set_ylim([0.01, 50])
            axs[0, i].set_yscale("log")
            axs[0, i].set_xscale("log")
        axs[0, i].set_xlabel("")
        axs[0, i].set_ylabel("")
        axs[0, i].legend(loc="best", prop={"size": 16}, frameon=True)
        axs[0, i].get_legend().set_title(f"{cat_label}")

        un_rel_diffs = np.nan_to_num((n - u) / n)
        axs[1, i].plot(
            bin_centers,
            un_rel_diffs,
            color=colors["up"],
            linestyle="none",
            marker="^",
            lw=0,
        )
        axs[1, i].axhline(y=0, color="black", alpha=0.5, linestyle="--")
        dn_rel_diffs = np.nan_to_num((n - d) / n)
        axs[2, i].errorbar(
            bin_centers,
            dn_rel_diffs,
            color=colors["down"],
            linestyle="none",
            marker="v",
            lw=0,
        )
        axs[2, i].axhline(y=0, color="black", alpha=0.5, linestyle="--")

        axs[1, i].set_ylim([-0.1, 0.1])
        axs[2, i].set_ylim([-0.1, 0.1])
        axs[1, i].set_ylabel("(Nom-Up)/Nom", fontsize=20)
        axs[2, i].set_ylabel("(Nom-Down)/Nom", fontsize=20)
        labels = {"raw": "Raw", "corr": "Corrected", "cons": "Constrained"}
        axs[2, i].set_xlabel(f"{labels[mass_type]} " + mass_label)

    for i in range(2):
        axs[0, i].set_ylim(axs[0, 2].get_ylim())
    plt.tight_layout()
    if outfile is not None:
        plt.savefig(outfile, format="pdf", dpi=800)
    plt.show()


def plot_systematic(nom, up, down, syst, cat_label, var_label):
    fig, axs = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(10, 16),
        dpi=200,
        gridspec_kw={"height_ratios": (4, 1, 1)},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.07)
    colors = {"up": "#45ADA8", "down": "#FF4E50", "nom": "#2A363B"}
    n, u, d = nom.values(), up.values(), down.values()
    bin_centers = nom.axes[0].centers
    # un_ratio, un_err = np.nan_to_num(n / u), ratio_uncertainty(n, u, "poisson")
    # dn_ratio, dn_err = np.nan_to_num(n / d), ratio_uncertainty(n, d, "poisson")

    up.plot1d(ax=axs[0], label=f"{syst} up", histtype="step", color=colors["up"])
    down.plot1d(ax=axs[0], label=f"{syst} down", histtype="step", color=colors["down"])
    nom.plot1d(ax=axs[0], label=f"{syst} nom", histtype="step", color=colors["nom"])

    axs[0].set_xlabel("")
    axs[0].set_ylabel("")
    axs[0].legend(loc="best", prop={"size": 16}, frameon=True)
    axs[0].get_legend().set_title(f"{cat_label}")

    un_rel_diffs = np.nan_to_num((n - u) / n)
    axs[1].plot(
        bin_centers,
        un_rel_diffs,
        color=colors["up"],
        linestyle="none",
        marker="^",
        lw=0,
    )
    dn_rel_diffs = np.nan_to_num((n - d) / n)
    axs[2].errorbar(
        bin_centers,
        dn_rel_diffs,
        color=colors["down"],
        linestyle="none",
        marker="v",
        lw=0,
    )
    axs[1].set_ylim([-0.1, 0.1])
    axs[2].set_ylim([-0.1, 0.1])
    axs[1].set_ylabel("(Nom-Up)/Nom", fontsize=20)
    axs[2].set_ylabel("(Nom-Down)/Nom", fontsize=20)
    axs[2].set_xlabel(var_label)

    plt.tight_layout()
    plt.show()
