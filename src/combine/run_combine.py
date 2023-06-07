from __future__ import annotations

import os
import sys

combine_cards = True
per_category = False

channel, years = sys.argv[1], sys.argv[2:]  # ["2017", "2018"]
print(years)
step1 = ["225", "250", "275", "300"]
step2 = ["325", "350", "375"]
step3 = ["400", "450"]
step4 = ["500", "600", "700"]
step5 = ["750", "800", "900", "1000", "1200", "1400"]
step6 = ["1600", "1800", "2000"]
mass_points = [
    "225",
    "250",
    "275",
    "300",
    "325",
    "350",
    "375",
    "400",
    "450",
    "500",
    "600",
    "700",
    "750",
    "800",
    "900",
    "1000",
    "1200",
    "1400",
    "1600",
    "1800",
    "2000",
]
channels = [channel]
categories = ["eeem", "eeet", "eemt", "eett", "mmem", "mmet", "mmmt", "mmtt"]

if combine_cards:
    for channel in channels:
        if per_category:
            for cat in categories:
                for mp in mass_points:
                    merge_cards = "combineCards.py "
                    for year in years:
                        merge_cards = (
                            merge_cards
                            + f"UL_{year}/azh_{year}_{channel}_{cat}_{mp}.txt "
                        )
                    merge_cards = merge_cards + f"> azh_run2_{channel}_{cat}_{mp}.txt"
                    print(merge_cards)
                    os.system(merge_cards)
                    os.system("mv azh_*.txt .datacards/")
        else:
            for mp in mass_points:
                merge_cards = "combineCards.py "
                for year in years:
                    merge_cards = (
                        merge_cards + f"UL_{year}/azh_{year}_{channel}_{mp}.txt "
                    )
                merge_cards = merge_cards + f"> azh_run2_{channel}_{mp}.txt"
                print(merge_cards)
                os.system(merge_cards)
                os.system("mv azh_*.txt .datacards/")

name = "all" if len(years) > 1 else years[0]
steps = [step1, step2, step3, step4, step5, step6]
rmax = [30, 25, 20, 15, 10, 20]
if not per_category:
    for channel in channels:
        for step in steps:
            for i, istep in enumerate(step):
                combine_cmd = (
                    "combine -M AsymptoticLimits --noFitAsimov --rMin=0 "
                    + f"--run blind --rMax={rmax[i]} --X-rtd MINIMIZER_analytic "
                    + "--cminDefaultMinimizerStrategy=0 --cminDefaultMinimizerTolerance=0.01 "
                    + f".datacards/azh_run2_{channel}_{istep}.txt "
                    + f"-t -1 -m {istep} -n .{name}_{channel}"
                )
                print(combine_cmd)
                os.system(combine_cmd)

else:
    for channel in channels:
        for cat in categories:
            print(cat)
            rmax1 = (
                "100"
                if cat == "eeem" or cat == "mmem"
                else "80"
                if cat == "eeet" or cat == "mmet"
                else "70"
                if cat == "eemt" or cat == "mmmt"
                else "60"
                if cat == "eett" or cat == "mmtt"
                else 30
            )
            rmax2 = (
                "80"
                if cat == "eeem" or cat == "mmem"
                else "70"
                if cat == "eeet" or cat == "mmet"
                else "60"
                if cat == "eemt" or cat == "mmmt"
                else "50"
                if cat == "eett" or cat == "mmtt"
                else 25
            )
            rmax3 = (
                "60"
                if cat == "eeem" or cat == "mmem"
                else "50"
                if cat == "eeet" or cat == "mmet"
                else "40"
                if cat == "eemt" or cat == "mmmt"
                else "35"
                if cat == "eett" or cat == "mmtt"
                else 20
            )
            rmax4 = (
                "40"
                if cat == "eeem" or cat == "mmem"
                else "35"
                if cat == "eeet" or cat == "mmet"
                else "30"
                if cat == "eemt" or cat == "mmmt"
                else "25"
                if cat == "eett" or cat == "mmtt"
                else 15
            )
            rmax5 = (
                "30"
                if cat == "eeem" or cat == "mmem"
                else "25"
                if cat == "eeet" or cat == "mmet"
                else "20"
                if cat == "eemt" or cat == "mmmt"
                else "20"
                if cat == "eett" or cat == "mmtt"
                else 10
            )
            rmax6 = (
                "50"
                if cat == "eeem" or cat == "mmem"
                else "45"
                if cat == "eeet" or cat == "mmet"
                else "40"
                if cat == "eemt" or cat == "mmmt"
                else "40"
                if cat == "eett" or cat == "mmtt"
                else 20
            )

            rmax = [rmax1, rmax2, rmax3, rmax4, rmax5, rmax6]
            for step in steps:
                for i, istep in enumerate(step):
                    combine_cmd = (
                        f"combine -M AsymptoticLimits --noFitAsimov --rMin=0 --run blind --rMax={rmax[i]}"
                        + " --X-rtd MINIMIZER_analytic --cminDefaultMinimizerStrategy=0"
                        + " --cminDefaultMinimizerTolerance=0.01 "
                        + f".datacards/azh_run2_{channel}_{cat}_{istep}.txt"
                        f" -t -1 -m {istep} -n .{year}_{channel}_{cat}"
                    )
                    os.system(combine_cmd)
                    print(f"finished running on mA={istep} GeV, {cat}.")
