import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tkinter import *
import interactive_plot
import pandas as pd
import numpy as np
import sys
import torch
from torch.utils.data import DataLoader

sys.path.append("..")
from model.CRNN import CRNN
from utils.config import *
from utils.filepath import *
from utils.DesedDataset import DesedDataset
from utils.utils import compute_plot_df, compute_prediction, compute_gradient_df, psds_f_score, confusion_df, gs_to_hex
import interactive_plot

def on_click(i,j, bgrid, options, cmap, cdf, gradient_df, max_duration, n_cluster, max_sample, fig, ax):
    # update button color
    for k in range(bgrid.shape[0]):
        for l in range(bgrid.shape[1]):
            bgrid[k][l].configure(bg=cmap[k][l])        
    bgrid[i][j].configure(bg="red")
    
    # update graph
    fig.clf()
    ax = fig.add_subplot(111)
    plot_df = compute_plot_df(cdf[options[i]][options[j]], n_cluster, max_sample, sample_rate, n_window, hop_size, n_mels, mel_min_max_freq, batch_size, max_duration=max_duration, gradient_df=gradient_df)
    interactive_plot.plot(plot_df, fig=fig, ax=ax)

def on_closing():
    plt.close()
    window.destroy()

if __name__ == "__main__":

    # load model
    model_df = torch.load(load_model_path, map_location="cpu")
    model = CRNN(*model_df["model"]["args"], **model_df["model"]["kwargs"])
    model.load_state_dict(model_df["model"]["state_dict"])

    # load datasets
    evaluation_df = pd.read_csv(evaluation_tsv_path, header=0, sep="\t")

    # load the list of labels of the evaluation dataset
    label_list = evaluation_df.event_label.dropna().sort_values().unique().tolist()

  
    # define dataloader for the valid synthetic dataset
    eval_dataset = DesedDataset(
        evaluation_df,
        label_list,
        save_evaluation_feature_path,
        sample_rate,
        n_window,
        hop_size,
        n_mels,
        mel_min_max_freq,
        max_frames,
        n_frames,
        pooling_time_ratio,
        model_df["scaler"]["state_dict"]["mean_"],
        model_df["scaler"]["state_dict"]["mean_of_square_"],
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=batch_size, drop_last=False, num_workers=n_workers
    )

    # compute prediction
    prediction = compute_prediction(
        model,
        eval_dataloader,
        label_list,
        pooling_time_ratio,
        median_window,
        sample_rate,
        hop_size,
        max_len_seconds,
    )

    # compute gradients
    gradient_df = compute_gradient_df(model, eval_dataloader)

    # evaluate metrics
    cdf = confusion_df(prediction, eval_dataloader.dataset.df, dtc_threshold, gtc_threshold, cttc_threshold)
    f_score = psds_f_score(cdf, evaluation_df)
    options = list(cdf.keys())
    n_option = len(options)

    # define GUI
    window = Tk()

    fig = Figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
  
    max_duration = IntVar(window)
    n_cluster = IntVar(window)
    max_sample = IntVar(window)

    max_duration.set(150)
    n_cluster.set(4)
    max_sample.set(30)

    fleft = Frame(window)
    fleft.pack(fill=Y, side=LEFT, expand=False)
    fright = Frame(window)
    fright.pack(fill=BOTH, side=RIGHT, expand=True)

    Label(fleft, text="Confusion matrix").pack()
    Label(fleft, text="(rows = predictions, columns = groundtruth)").pack()
    fgrid = Frame(fleft, width=500, height=500)
    fgrid.pack()


    for i in range(n_option):
        Label(
            fgrid, 
            text=options[i][:15], 
            height=1,
            width=15
        ).grid(row=i+1, column=0)
        Label(fgrid, text=options[i][0], height=1, width=3).grid(row=0, column=i+1)

    
    cm = [[len(cdf[options[i]][options[j]]) for j in range(n_option)] for i in range(n_option)]
    alpha = 150/(np.max(cm)+1e-6)
    cmap = [[gs_to_hex(255-int(alpha*cm[i][j])) for j in range(n_option)] for i in range(n_option)]
    bgrid = np.empty((n_option, n_option), dtype=object)
    for i in range(n_option):
        for j in range(n_option):
            bgrid[i][j] = Button(
                fgrid, 
                text=str(cm[i][j]), 
                bg=cmap[i][j],
                height=1,
                width=3,
                command=lambda i=i, j=j:on_click(i,j, bgrid, options, cmap, cdf, gradient_df, max_duration.get(), n_cluster.get(), max_sample.get(), fig, ax)
            )
            bgrid[i][j].grid(row=i+1, column=j+1)

    Label(fleft,  text="\nf-score: " + str(f_score)).pack()
    Label(fleft, text="Number of clusters").pack()
    Scale(fleft, from_=0, to=30, length=500, orient=HORIZONTAL, variable=n_cluster, command=lambda v:n_cluster.set(v)).pack()
    Label(fleft, text="Max number of samples").pack()
    Scale(fleft, from_=0, to=1000, length=500, orient=HORIZONTAL, variable=max_sample, command=lambda v:max_sample.set(v)).pack()
    Label(fleft, text="Duration").pack()
    Scale(fleft, from_=0, to=10000, length=500, orient=HORIZONTAL, variable=max_duration, command=lambda v:max_duration.set(v)).pack()

    canvas = FigureCanvasTkAgg(fig, master=fright)
    canvas.get_tk_widget().pack(fill=BOTH, expand=True)

    window.protocol("WM_DELETE_WINDOW", on_closing)
    window.mainloop()
