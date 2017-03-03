import pandas as pd
import numpy as np

from sklearn.model_selection import ShuffleSplit
from importlib import import_module

def score_function(y_true, y_pred):
    '''Compute a clustering score.

    Parameters
    ----------
    y_true : np.array, shape = (n, 2)
        The ground truth.
        first column: event_id
        second column: cluster_id
    y_pred : np.array, shape = (n, 2)
        The predicted cluster assignment.
        first column: event_id
        second column: predicted cluster_id
    """
    '''
    score = 0.
    event_ids = y_true[:, 0]
    y_true_cluster_ids = y_true[:, 1]
    y_pred_cluster_ids = y_pred[:, 1]

    # loop over events
    unique_event_ids = np.unique(event_ids)
    for event_id in unique_event_ids:
        efficiency_total = 0.

        # indices of the hits in this event
        event_indices = (event_ids==event_id)

        # assingments and particle ids of these hits
        cluster_ids_true = y_true_cluster_ids[event_indices]
        cluster_ids_pred = y_pred_cluster_ids[event_indices]

        # the assignment ids, each of which will be assigned to a particle id
        unique_cluster_ids = np.unique(cluster_ids_true)
        n_cluster = len(unique_cluster_ids)
        n_sample = len(cluster_ids_true)

        assigned_cluster = np.full(
            shape=n_cluster, fill_value=-1, dtype='int64')
        point_in_cluster = np.full(
            shape=n_cluster, fill_value=0, dtype='int64')
        efficiency = np.full(shape=n_cluster, fill_value=0.)

        # assign points to unique_cluster_ids
        for i, cluster_id in enumerate(unique_cluster_ids):
            efficiency[i] = 0.

            # the hits belonging to the particle
            true_points = cluster_ids_true[cluster_ids_true == cluster_id]
            # the assignments of the same hits
            found_points = cluster_ids_pred[cluster_ids_true == cluster_id]

            # find the biggest cluster within the hits of the particle
            n_sub_cluster = len(np.unique(found_points[found_points >= 0]))
            if(n_sub_cluster > 0):
                b = np.bincount(
                    (found_points[found_points >= 0]).astype(dtype='int64'))
                a = np.argmax(b)
                maxcluster = a
                assigned_cluster[i] = maxcluster
                point_in_cluster[i] = len(
                    found_points[found_points == maxcluster])

        # loop over particles to measure what fraction is good
        sorted = np.argsort(point_in_cluster)
        point_in_cluster = point_in_cluster[sorted]
        assigned_cluster = assigned_cluster[sorted]
        i = 0
        for cluster_id in unique_cluster_ids:
            i_point = assigned_cluster[i]
            # if there is another particle with bigger overlap with the
            # same cluster, drop this particle
            if i_point < 0 or\
                    len(assigned_cluster[assigned_cluster == i_point]) > 1:
                point_in_cluster = np.delete(point_in_cluster, i)
                assigned_cluster = np.delete(assigned_cluster, i)
            else:
                i += 1
        n_good = 0.
        # sum the remaining hits for the good particles
        n_good = np.sum(point_in_cluster)
        efficiency_total = efficiency_total + 1. * n_good / n_sample
        score += efficiency_total
    score /= len(event_ids)
    return efficiency_total


filename = 'public_train.csv'


def read_data(filename):
    df = pd.read_csv(filename)
    y_df = df.drop(['layer', 'iphi', 'x', 'y'], axis=1)
    X_df = df.drop(['particle_id'], axis=1)
    return X_df.values, y_df.values


def train_submission(module_path, X_array, y_array, train_is):
    clusterer = import_module('clusterer', module_path)
    cls = clusterer.Clusterer()
    cls.fit(X_array[train_is], y_array[train_is])
    return cls


def test_submission(trained_model, X_array, test_is):
    cls = trained_model
    y_pred = cls.predict(X_array[test_is])
    return np.stack(
        (X_array[test_is][:, 0], y_pred), axis=-1).astype(dtype='int')


# We do a single fold because blending would not work anyway:
# mean of cluster_ids make no sense
def get_cv(y_train_array):
    unique_event_ids = np.unique(y_train_array[:, 0])
    event_cv = ShuffleSplit(n_splits=1, test_size=0.5, random_state=57)
    for train_event_is, test_event_is in event_cv.split(unique_event_ids):
        train_is = np.where(np.in1d(y_train_array[:, 0], train_event_is))
        test_is = np.where(np.in1d(y_train_array[:, 0], test_event_is))
        yield train_is, test_is


if __name__ == '__main__':
    print("Reading file ...")
    X, y = read_data(filename)
    unique_event_ids = np.unique(X[:, 0])
    cv = get_cv(y)
    print("Training ...")
    for train_is, test_is in cv:
        trained_model = train_submission('', X, y, train_is)
        y_pred = test_submission(trained_model, X, test_is)
        score = score_function(y[test_is], y_pred)
        print 'score = ', score

