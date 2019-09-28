import pickle


def save_dict(d, path):
    with open(path, 'wb') as fout:
        pickle.dump(d, fout)

def load_dict(path):
    with open(path, 'rb') as fin:
        return pickle.load(fin)