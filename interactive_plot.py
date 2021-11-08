import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.backend_bases import MouseButton
from tkinter.filedialog import asksaveasfilename
import winsound
import cv2


# interactive plot of a dataframe
# the dataframe is supposed to have the following columns
# df["x"]: x coordinate of points to draw
# df["y"]: y coordinate of points to draw
# df["im"]: images corresponding to points
# df["audio"]: sound corresponding to points
# df["label"]: label corresponding to points
def plot(df, fig=None, ax=None):

    if fig==None or ax==None:
        fig, ax = plt.subplots()

    
    label_list = df["label"].unique().tolist()
    sc = ax.scatter(
        df["x"],
        df["y"],
        c=list(map(lambda x: label_list.index(x), df["label"].tolist())),
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
        "im_scale":100,
        "range":max(ax.get_xlim()[1]-ax.get_xlim()[0], ax.get_ylim()[1]-ax.get_ylim()[0]),
        "ab": ab,
        "label": None,
        "show_spec": True,
    }
    fig.canvas.mpl_connect("motion_notify_event", lambda event: _hover(state, event))
    fig.canvas.mpl_connect("button_press_event", lambda event: _click(state, event))
    fig.canvas.mpl_connect("scroll_event", lambda event: _scroll(state, event))
    fig.canvas.draw()


def _play_sound(state, label):
    winsound.PlaySound(
        state["df"].iloc[label]["audio"], winsound.SND_ASYNC | winsound.SND_ALIAS
    )

def _stop_sound():
    winsound.PlaySound(None, winsound.SND_PURGE)


def _update_im(state, label):
    # show either spectrogram or saliency map
    if not state["show_spec"] and state["df"].iloc[label]["gradient"] is not None:
        im = state["df"].iloc[label]["gradient"] 
    else:
        im = state["df"].iloc[label]["spectrogram"] 
    
    im_resized = cv2.resize(im, dsize=(state["im_scale"]*2, state["im_scale"]), interpolation=cv2.INTER_CUBIC)
    im_normalized = cv2.normalize(im_resized, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    state["im"].set_data(im_normalized)
    state["fig"].canvas.draw_idle()

def _show_image(state, event, label):
    # prepare plot
    w, h = state["fig"].get_size_inches() * state["fig"].dpi
    ws = (event.x > w / 2.0) * -1 + (event.x <= w / 2.0)
    hs = (event.y > h / 2.0) * -1 + (event.y <= h / 2.0)
    state["ab"].xybox = (50.0 * ws, 50.0 * hs)
    state["ab"].set_visible(True)
    state["ab"].xy = (state["df"].iloc[label]["x"], state["df"].iloc[label]["y"])

    # update title
    state["ax"].set_title(state["df"].iloc[label]["audio"])

    # set the image corresponding to that point
    _update_im(state, label)

def _hide_image(state):
    state["ab"].set_visible(False)
    state["fig"].canvas.draw_idle()

def _save_point(state, label):
    
    # plot
    if state["df"].iloc[label]["gradient"] is not None:
        fig, ax = plt.subplots(nrows=2)
        fig.suptitle(state["df"].iloc[label]["audio"])
        ax[0].imshow(state["df"].iloc[label]["spectrogram"])
        ax[1].plot(state["df"].iloc[label]["gradient"])
    else:
        fig, ax = plt.subplots()
        ax.set_title(state["df"].iloc[label]["audio"])
        ax.imshow(state["df"].iloc[label]["spectrogram"])

    # ask filepath and save to file
    filepath = asksaveasfilename(defaultextension=".jpg")
    fig.savefig(filepath,bbox_inches='tight')
    #plt.close(fig)

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
                state["show_spec"] = not state["show_spec"]
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
                state["im_scale"] = state["im_scale"] + 50
            else:
                state["im_scale"] = max(state["im_scale"] - 50, 50)
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
    