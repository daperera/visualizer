import numpy as np
import openl3
import soundfile as sf
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from itertools import repeat, chain, islice

def compute_plot_df(filepath_list, display_df, n_cluster, max_sample, max_duration, batch_size, embedding_type="openl3"):  

    # generate subset of rows
    sample_list = np.random.choice(range(len(filepath_list)), min(len(filepath_list), max_sample), replace=False)
    filepath_sample = [filepath_list[sample] for sample in sample_list]

    # select subset
    df = display_df.loc[display_df.filepath.isin(filepath_sample)]

    # extract audio from file
    audio_list, sr_list = [], []
    for filepath in filepath_sample:
        # load audio file
        audio, sr = sf.read(filepath)
        audio = list(trimmer(audio, max_duration))  
        audio_list.append(np.asarray(audio))
        sr_list.append(sr)
    df["__audio__"] = audio_list

    # compute openl3 embedding
    if embedding_type == "openl3":
        embedding, _ = openl3.get_audio_embedding(audio_list, sr_list, batch_size=batch_size)
        input = np.array(list(map(lambda x : np.reshape(x, (x.shape[0]*6144)), embedding)))
    else:
        input = TSNE(n_components=2, learning_rate="auto", init="random").fit(audio_list)

    # run K mean
    n_cluster = min(n_cluster, len(filepath_list))
    kmeans = KMeans(n_clusters=n_cluster, random_state=0, max_iter=1000).fit(
        input
    )

    # project to 2d space using PCA
    if len(input) > 1:
        data_2d = PCA(n_components=2).fit_transform(input)
    else:
        data_2d = [[(0,0)]]

    df["__x__"] = data_2d[:, 0]
    df["__y__"] = data_2d[:, 1]
    df["__label__"] = kmeans.labels_

    return df

def trimmer(seq, size, filler=0):
    return islice(chain(seq, repeat(filler)), size)

def rgb_to_hex(rgb):
    return "#%02x%02x%02x" % rgb

def gs_to_hex(gs):
    return rgb_to_hex((gs,gs,gs))

def color_map(confusion_matrix):
    n_options = len(confusion_matrix)
    alpha = 150/(np.max(confusion_matrix)+1e-6)
    cmap = [[gs_to_hex(255-int(alpha*confusion_matrix[i][j])) for j in range(n_options)] for i in range(n_options)]
    return cmap