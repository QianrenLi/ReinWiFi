CLUSTER_KERNEL = None
# Read data from env
import csv
class csvReader():
    @staticmethod
    def read(path):
        with open(path, "r") as f:
            reader = csv.reader(f)
            data = list(reader)
        return data
    @staticmethod
    def _str_to_float(data) -> list:
        return [float(d) for d in data]
    @staticmethod
    def _get_action_num(data):
        return len(data[1])
    @staticmethod
    def _get_state_num(data):
        return len(data[0])
    @staticmethod
    def _interpreter(data, idx):
        return csvReader._str_to_float(data[idx])
    @staticmethod
    def _action_list(data):
        return [csvReader._interpreter(data, idx) for idx in range(1, len(data), 2)]
    @staticmethod
    def _state_list(data):
        return [csvReader._interpreter(data, idx) for idx in range(0, len(data), 2)]
    @staticmethod
    def get_data_vector(path):
        _data = csvReader.read(path)
        action_list = csvReader._action_list(_data)
        state_list = csvReader._state_list(_data)
        ##
        for idx in range(len(state_list)):
            state_list[idx][-1] = state_list[idx][-1] / 10
        ##
        assert(len(action_list) == len(state_list))
        ## cascade action and state in one line
        data = [action_list[idx] + state_list[idx] for idx in range(len(action_list))]
        return data

def k_means(data, k):
    import sklearn
    from sklearn.cluster import KMeans, Birch, AgglomerativeClustering, DBSCAN, SpectralClustering
    from sklearn.mixture import GaussianMixture
    import numpy as np
    print(np.array(data).shape)
    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(np.array(data))
    # kmeans = Birch(n_clusters=k).fit(np.array(data))
    # kmeans = SpectralClustering(n_clusters=k, assign_labels="discretize",random_state=0).fit(np.array(data))
    # kmeans = AgglomerativeClustering(n_clusters=k).fit(np.array(data))
    # kmeans = DBSCAN(eps=0.3, min_samples=10).fit(np.array(data))
    # kmeans = GaussianMixture(n_components=k).fit(np.array(data))
    return kmeans

def read_data(idx, idxRange):
    data = None
    for id in idxRange:
        path = "./env/1/env_t{}_{}.csv".format(idx, id)
        _data = csvReader.get_data_vector(path)
        data = data + _data if data is not None else _data
    return data

def read_label(idx, idxRange ,require_label = False):
    data = None
    # test_label = {1: "Low\nInterference", 2: "Medium\nInterference", 3: "High\nInterference"}
    test_label = {}
    [test_label.update({j:f'{i}'}) for i,j in enumerate(idxRange)]
    for id in idxRange:
        path = "./env/1/env_t{}_{}.csv".format(idx, id)
        _data = csvReader.get_data_vector(path)
        _label = [id] * len(_data)
        _test_label = [test_label[id]] * len(_data)
        if require_label:
            data = data + _test_label if data is not None else _test_label
        else:
            data = data + _label if data is not None else _label
    return data

def test_k_means(data, kmeans):
    import numpy as np
    try:
        res = kmeans.predict(np.array(data))
        return res
    except:
        res = kmeans.fit_predict(np.array(data))
        return res
