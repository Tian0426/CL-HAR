import os
import logging
import sys
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
import seaborn as sns
import matplotlib.pyplot as plt

def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger

def tsne(latent, y_ground_truth, save_dir):
    """
        Plot t-SNE embeddings of the features
    """
    latent = latent.cpu().detach().numpy()
    # y_ground_truth = y_ground_truth.cpu().detach().numpy()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(latent)
    plt.figure(figsize=(16,10))
    set_y = set(y_ground_truth)
    num_labels = len(set_y)
    sns_plot = sns.scatterplot(
        x=tsne_results[:,0], y=tsne_results[:,1],
        hue=y_ground_truth,
        palette=sns.color_palette("hls", num_labels),
        legend="full",
        alpha = 0.5
        )

    sns_plot.get_figure().savefig(save_dir)


def mds(latent, y_ground_truth, save_dir):
    """
        Plot MDS embeddings of the features
    """
    latent = latent.cpu().detach().numpy()
    mds = MDS(n_components=2)
    mds_results = mds.fit_transform(latent)
    plt.figure(figsize=(16,10))
    set_y = set(y_ground_truth)
    num_labels = len(set_y)
    sns_plot = sns.scatterplot(
        x=mds_results[:,0], y=mds_results[:,1],
        hue=y_ground_truth,
        palette=sns.color_palette("hls", num_labels),
        # data=df_subset,
        legend="full",
        alpha=0.5
        )

    sns_plot.get_figure().savefig(save_dir)