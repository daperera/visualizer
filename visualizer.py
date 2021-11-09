import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from tkinter import *
import numpy as np

from . import interactive_plot
from .utils import *

class Visualizer:

    def __init__(self, confusion_df, display_df):
        self.cdf = confusion_df
        self.display_df = display_df
        self.options = list(self.cdf.keys())

    def _on_click(self, i,j):
        # update button color
        for k in range(self.bgrid.shape[0]):
            for l in range(self.bgrid.shape[1]):
                self.bgrid[k][l].configure(bg=self.cmap[k][l])        
        self.bgrid[i][j].configure(bg="red")
        
        # update graph
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        plot_df = compute_plot_df(
            self.cdf[self.options[i]][self.options[j]], 
            self.display_df,
            self.n_cluster.get(), 
            self.max_sample.get(), 
            self.max_duration.get(),
            self.batch_size.get(), 
        )
        interactive_plot.plot(plot_df, fig=self.fig, ax=self.ax)

    def _on_closing(self):
        plt.close()
        self.window.destroy()

    def show(self):

        # define GUI
        self.window = Tk()

        self.fig = Figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111)
    
        self.max_duration = IntVar(self.window)
        self.n_cluster = IntVar(self.window)
        self.max_sample = IntVar(self.window)
        self.batch_size = IntVar(self.window)

        self.max_duration.set(150)
        self.n_cluster.set(4)
        self.max_sample.set(30)
        self.batch_size.set(24)

        fleft = Frame(self.window)
        fleft.pack(fill=Y, side=LEFT, expand=False)
        fright = Frame(self.window)
        fright.pack(fill=BOTH, side=RIGHT, expand=True)


        # building confusion matrix interface
        Label(fleft, text="Confusion matrix").pack()
        Label(fleft, text="(rows = predictions, columns = groundtruth)").pack()
        fgrid = Frame(fleft, width=500, height=500)
        fgrid.pack()

        # building confusion matrix titles
        n_option = len(self.options)
        for i in range(n_option):
            Label(
                fgrid, 
                text=self.options[i][:15], 
                height=1,
                width=15
            ).grid(row=i+1, column=0)
            Label(fgrid, text=self.options[i][0], height=1, width=3).grid(row=0, column=i+1)

        # building confusion matrix grid
        cm = [[len(self.cdf[self.options[i]][self.options[j]]) for j in range(n_option)] for i in range(n_option)]
        self.cmap = color_map(cm)
        self.bgrid = np.empty((n_option, n_option), dtype=object)
        for i in range(n_option):
            for j in range(n_option):
                self.bgrid[i][j] = Button(
                    fgrid, 
                    text=str(cm[i][j]), 
                    bg=self.cmap[i][j],
                    height=1,
                    width=3,
                    command=lambda i=i, j=j:self._on_click(i,j)
                )
                self.bgrid[i][j].grid(row=i+1, column=j+1)

        # interface to select visualization parameters
        Label(fleft, text="\nNumber of clusters").pack()
        Scale(fleft, from_=0, to=30, length=500, orient=HORIZONTAL, variable=self.n_cluster, command=lambda v:self.n_cluster.set(v)).pack()
        Label(fleft, text="Max number of samples").pack()
        Scale(fleft, from_=0, to=1000, length=500, orient=HORIZONTAL, variable=self.max_sample, command=lambda v:self.max_sample.set(v)).pack()
        Label(fleft, text="Duration").pack()
        Scale(fleft, from_=0, to=10000, length=500, orient=HORIZONTAL, variable=self.max_duration, command=lambda v:self.max_duration.set(v)).pack()
        Label(fleft, text="Batch size").pack()
        Scale(fleft, from_=0, to=100, length=500, orient=HORIZONTAL, variable=self.batch_size, command=lambda v:self.batch_size.set(v)).pack()

        canvas = FigureCanvasTkAgg(self.fig, master=fright)
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)

        self.window.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.window.mainloop()
