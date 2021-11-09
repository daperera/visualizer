import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.backend_bases import MouseButton
from tkinter.filedialog import asksaveasfilename
import winsound
import cv2



def plot(df, fig, ax):
    
    label_list = df["__label__"].unique().tolist()
    sc = ax.scatter(
        df["__x__"],
        df["__y__"],
        c=list(map(lambda x: label_list.index(x), df["__label__"].tolist())),
        s=100,
        cmap=plt.cm.RdYlGn,
        norm=plt.Normalize(1, 4),
    )
   
    im = OffsetImage([[0]], zoom=1) 

    ab = AnnotationBbox(
        im,
        (0, 0),
        xybox=(50.0, 50.0),
        xycoords="data",
        boxcoords="offset points",
        pad=0.3,
        arrowprops=dict(arrowstyle="->"),
    )
    ax.add_artist(ab)
    ab.set_visible(False)
    ab.set_zorder(10)

    state = {
        "df": df,
        "fig": fig,
        "ax": ax,
        "sc": sc,
        "im": im,
        "zoom":100,
        "range":max(ax.get_xlim()[1]-ax.get_xlim()[0], ax.get_ylim()[1]-ax.get_ylim()[0]),
        "ab": ab,
        "label": None,
        "key_list": list([key for key in df.keys() if key not in ["__x__", "__y__", "__audio__", "__label__", "filepath"]]),
        "im_key": 0,
    }
    fig.canvas.mpl_connect("motion_notify_event", lambda event: _hover(state, event))
    fig.canvas.mpl_connect("button_press_event", lambda event: _click(state, event))
    fig.canvas.mpl_connect("scroll_event", lambda event: _scroll(state, event))
    fig.canvas.draw()


def _play_sound(state, label):
    winsound.PlaySound(
        state["df"].iloc[label]["__audio__"], winsound.SND_ASYNC | winsound.SND_ALIAS
    )

def _stop_sound():
    winsound.PlaySound(None, winsound.SND_PURGE)


def _update_im(state, label):
    # get image from state and prepare it
    print(state["im_key"])
    print(state["key_list"])
    im = state["df"].iloc[label][state["key_list"][state["im_key"]]]
    im_resized = cv2.resize(im, dsize=(state["zoom"]*2, state["zoom"]), interpolation=cv2.INTER_CUBIC)
    im_normalized = cv2.normalize(im_resized, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    # update plot
    state["im"].set_data(im_normalized)
    state["fig"].canvas.draw_idle()

    # update title
    state["ax"].set_title(state["key_list"][state["im_key"]])

def _show_image(state, event, label):
    # prepare plot
    w, h = state["fig"].get_size_inches() * state["fig"].dpi
    ws = (event.x > w / 2.0) * -1 + (event.x <= w / 2.0)
    hs = (event.y > h / 2.0) * -1 + (event.y <= h / 2.0)
    state["ab"].xybox = (50.0 * ws, 50.0 * hs)
    state["ab"].set_visible(True)
    state["ab"].xy = (state["df"].iloc[label]["__x__"], state["df"].iloc[label]["__y__"])

    # set the image corresponding to that point
    _update_im(state, label)

def _hide_image(state):
    state["ab"].set_visible(False)
    state["fig"].canvas.draw_idle()

def _save_point(state, label):
    n_keys = len(state["key_list"])
    if n_keys == 1:
        fig, ax = plt.subplots()
        ax.set_title(state["df"].iloc[label]["audio"])
        ax.imshow(state["df"].iloc[label][state["key_list"][0]])
    else:
        fig, ax = plt.subplots(nrows=n_keys)
        fig.suptitle(state["df"].iloc[label]["audio"])
        for i, key in enumerate(state["key_list"]): 
            ax[i].imshow(state["df"].iloc[label][key])
            ax[i].title(key)

    # ask filepath and save to file
    filepath = asksaveasfilename(defaultextension=".jpg")
    fig.savefig(filepath,bbox_inches='tight')

def _hover(state, event):
    if event.inaxes == state["ax"]:
        cont, ind = state["sc"].contains(event)
        if cont:
            if state["label"] == None:
                label = ind["ind"][0]
                state["label"] = label
                _show_image(state, event, label)
                

        elif state["label"] != None:
            state["label"] = None
            _hide_image(state)


def _click(state, event):
    # if left click on a point, print the corresponding filepath
    # if left click elsewhere, stop playing current sound
    # if right click, alternate between spectrogram and saliency view
    if event.inaxes == state["ax"]:
        cont, ind = state["sc"].contains(event) 
        if cont:
            label = ind["ind"][0]
            if event.button is MouseButton.LEFT:
                _play_sound(state, label)
            elif event.button is MouseButton.RIGHT:
                state["im_key"] = (state["im_key"] + 1) % len(state["key_list"])
                _update_im(state, label)
            else:
                _save_point(state, label)
        else:
            _stop_sound()

def _scroll(state, event):
    if event.inaxes == state["ax"]:
        cont, ind = state["sc"].contains(event)

        # if a point is hovered : update image scale
        if cont:
            if event.button == "up":
                state["zoom"] = state["zoom"] + 50
            else:
                state["zoom"] = max(state["zoom"] - 50, 50)
            _update_im(state, ind["ind"][0])
        # else zoom in the graph and center at the position of the mouse
        else:
            if event.button == "up":
                state["range"] = state["range"] * 0.9
            else:
                state["range"] = state["range"] * 1.1
            state["ax"].set_xlim(event.xdata - state["range"], event.xdata + state["range"])
            state["ax"].set_ylim(event.ydata - state["range"], event.ydata + state["range"])
            state["fig"].canvas.draw_idle()
    