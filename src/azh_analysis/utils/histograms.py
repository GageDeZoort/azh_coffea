from __future__ import annotations

import numpy as np
from hist import Hist
from hist.axis import IntCategory, Regular, StrCategory, Variable


def integrate(hist):
    bins = np.array(hist.axes[0])
    widths = bins[:, 1] - bins[:, 0]
    vals = hist.values()
    return sum(widths * vals)


def make_analysis_hist_stack(fileset, year):
    # bin variables along axes
    group_axis = StrCategory(
        name="group",
        categories=[],
        growth=True,
    )
    category_axis = StrCategory(
        name="category",
        categories=[],
        growth=True,
    )
    leg_axis = StrCategory(
        name="leg",
        categories=[],
        growth=True,
    )
    btags_axis = IntCategory(
        name="btags",
        categories=[0, 1],
        growth=True,
    )
    mass_type_axis = StrCategory(
        name="mass_type",
        categories=[],
        growth=True,
    )
    syst_shift_axis = StrCategory(
        name="syst_shift",
        categories=[],
        growth=True,
    )

    split_str = f"_{year}"
    pt = {
        dataset.split(split_str)[0]: Hist(
            group_axis,
            category_axis,
            btags_axis,
            syst_shift_axis,
            leg_axis,
            Regular(name="pt", bins=10, start=0, stop=200),
        )
        for dataset in fileset.keys()
    }
    met = {
        dataset.split(split_str)[0]: Hist(
            group_axis,
            category_axis,
            btags_axis,
            syst_shift_axis,
            Regular(name="met", bins=10, start=0, stop=200),
        )
        for dataset in fileset.keys()
    }
    mtt = {
        dataset.split(split_str)[0]: Hist(
            group_axis,
            category_axis,
            btags_axis,
            syst_shift_axis,
            mass_type_axis,
            Regular(name="mass", bins=20, start=0, stop=300),
        )
        for dataset in fileset.keys()
    }
    m4l_reg = {
        dataset.split(split_str)[0]: Hist(
            group_axis,
            category_axis,
            btags_axis,
            syst_shift_axis,
            mass_type_axis,
            Regular(name="mass", bins=50, start=0, stop=2500),
        )
        for dataset in fileset.keys()
    }
    m4l_fine = {
        dataset.split(split_str)[0]: Hist(
            group_axis,
            category_axis,
            btags_axis,
            syst_shift_axis,
            mass_type_axis,
            Regular(name="mass", bins=250, start=0, stop=2500),
        )
        for dataset in fileset.keys()
    }
    m4l = {
        dataset.split(split_str)[0]: Hist(
            group_axis,
            category_axis,
            btags_axis,
            syst_shift_axis,
            mass_type_axis,
            Variable(
                [
                    200,
                    220,
                    240,
                    260,
                    280,
                    300,
                    320,
                    340,
                    360,
                    380,
                    400,
                    450,
                    550,
                    700,
                    1000,
                    2400,
                ],
                name="mass",
            ),
        )
        for dataset in fileset.keys()
    }
    mll = {
        dataset.split(split_str)[0]: Hist(
            group_axis,
            category_axis,
            btags_axis,
            syst_shift_axis,
            Regular(name="mll", bins=10, start=60, stop=120),
        )
        for dataset in fileset.keys()
    }
    return {
        "mtt": mtt,
        "m4l": m4l,
        "m4l_reg": m4l_reg,
        "m4l_fine": m4l_fine,
        "mll": mll,
        "pt": pt,
        "met": met,
    }


def make_fr_hist_stack(fileset, year):
    group_axis = StrCategory(
        name="group",
        categories=[],
        growth=True,
    )
    category_axis = StrCategory(
        name="category",
        categories=[],
        growth=True,
    )
    prompt_axis = StrCategory(
        name="prompt",
        categories=[],
        growth=True,
    )
    numerator_axis = StrCategory(
        name="numerator",
        categories=[],
        growth=True,
    )
    eta_axis = StrCategory(
        name="eta_bin",
        categories=[],
        growth=True,
    )
    decay_mode_axis = IntCategory(
        [-1, 0, 1, 2, 10, 11, 15],
        name="decay_mode",
    )

    pt = {
        dataset.split(f"_{year}")[0]: Hist(
            group_axis,
            category_axis,
            prompt_axis,
            numerator_axis,
            decay_mode_axis,
            eta_axis,
            Regular(name="pt", bins=30, start=0, stop=300),
        )
        for dataset in fileset.keys()
    }
    met = {
        dataset.split(f"_{year}")[0]: Hist(
            group_axis,
            category_axis,
            prompt_axis,
            numerator_axis,
            decay_mode_axis,
            eta_axis,
            Regular(name="met", bins=30, start=0, stop=300),
        )
        for dataset in fileset.keys()
    }

    mll = {
        dataset.split(f"_{year}")[0]: Hist(
            group_axis,
            category_axis,
            prompt_axis,
            numerator_axis,
            decay_mode_axis,
            eta_axis,
            Regular(name="mll", bins=20, start=60, stop=120),
        )
        for dataset in fileset.keys()
    }

    mT = {
        dataset.split(f"_{year}")[0]: Hist(
            group_axis,
            category_axis,
            prompt_axis,
            numerator_axis,
            decay_mode_axis,
            eta_axis,
            Regular(name="mT", bins=30, start=0, stop=300),
        )
        for dataset in fileset.keys()
    }

    return {
        "mll": mll,
        "pt": pt,
        "met": met,
        "mT": mT,
    }
