import pickle
path = "/dataset.pkl"
with open(path, 'rb') as f:
    dataset = pickle.load(f)
    dataset_ids = [x[0] for x in dataset]
    print(dataset_ids)