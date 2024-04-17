import numpy as np
import pickle as pkl


COLOR_PALETTE = {"None": "#151D29",   # black
                "random": "#9AA7B1",  # grey
                "V1_shuffle": "#8E2D30",
                "TO_shuffle": "#2E59A7",
                "V1": "#8E2D30",
                "V2": "#EE7959",
                "V4": "#FAC03D",
                "LO": "#108B96",
                "TO":"#2E59A7",
                "VO": "magenta",
                "PHC": "purple"}

LINESTYLE = {"None": "dashed",
           "random": "dashed",
           "V1_shuffle": "dotted",
           "TO_shuffle": "dotted",
           "V1": "solid",
           "V2": "solid",
           "V4": "solid",
           "LO": "solid",
           "TO":"solid",
           "VO": "solid",
           "PHC": "solid"}

MARKERS = {"None": "o",
           "random": "o",
           "V1_shuffle": "d",
           "TO_shuffle": "d",
           "V1": "D",
           "V2": "D",
           "V4": "D",
           "LO": "D",
           "TO":"D",
           "VO": "D",
           "PHC": "D"}

PLOT_LB = {"None": "None",
           "random": "Random",
           "V1_shuffle": "V1-shuffle",
           "TO_shuffle": "TO-shuffle",
           "V1": "V1",
           "V2": "V2",
           "V4": "V4",
           "LO": "LO",
           "TO":"TO",
           "VO": "VO",
           "PHC": "PHC",
           "wd0.003": "WD-0.003",
           "wd0.004": "WD-0.004",
           "wd0.005": "WD-0.005",
           "wd0.007": "WD-0.007",
           "wd0.01": "WD-0.01"}


ROI_ORDER_LS = ["None", "random", "V1_shuffle", "TO_shuffle", "V1", "V2", "V4", "VO", "PHC", "LO", "TO"]


PLOT_LINF_THRES = [f"{k: .3f}".strip() for k in np.arange(0.001, 0.02, 0.002)]


def normalize_acc(init_acc, acc):
    return 1.0 - np.abs((np.asarray(acc) - init_acc) / init_acc)

def pickle_load(fpth):
    print(f"loading from: {fpth}")
    with open(fpth, 'rb') as f:
        return pkl.load(f)
