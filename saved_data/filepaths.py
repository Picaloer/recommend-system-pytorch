import os

ROOT = os.path.split(os.path.realpath(__file__))[0]


class Trained_para:
    PCA_PARA = os.path.join(ROOT, "pca_parameter.pkl")
    DCN_MODEL = os.path.join(ROOT, "dcn_model.pth")


class Saved_para:
    CO_MATRIX = os.path.join(ROOT, "co-occurrence_matrix.npy")
