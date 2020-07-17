import numpy as np


def loops_fill_row(arr):
    out = arr.copy()
    for row_idx in range(out.shape[0]):
        for col_idx in range(1, out.shape[1]):
            if out[row_idx, col_idx] == 0.:
                out[row_idx, col_idx] = out[row_idx, col_idx - 1]
    return out


def loops_fill_col(arr):
    out = arr.copy()
    for row_idx in range(1, out.shape[0]):
        for col_idx in range(out.shape[1]):
            if out[row_idx, col_idx] == 0.:
                out[row_idx, col_idx] = out[row_idx - 1, col_idx]
    return out


def impute_nans(arr):
    out = arr.copy()
    for row_idx in range(out.shape[0]):
        for col_idx in range(out.shape[1]):
            if np.isnan(out[row_idx, col_idx]):
                if col_idx == 0:
                    out[row_idx, col_idx] = out[row_idx - 1, col_idx]
                else:
                    out[row_idx, col_idx] = out[row_idx, col_idx - 1]
    return out


if __name__ == '__main__':
    f = open('features.txt', 'r')

    features = []
    for line in f.readlines():
        features.append([float(x) for x in line.strip().split()])
    
    features = np.asarray(features)
    preds = loops_fill_row(features)
    preds = loops_fill_col(preds)
    preds = impute_nans(preds)

    np.savetxt('preds.txt', preds)