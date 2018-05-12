import numpy as np
import tqdm


class BinaryFFMFormatter:
    def __init__(self, col_index_to_string):
        self.col_index_to_string = col_index_to_string
        self.num_cols = len(col_index_to_string)

    def format_row(self, row, label):
        strings = [str(label)]
        for col_index in row.indices:
            strings.append(self.col_index_to_string[col_index])
        string = " ".join(strings)
        return string

    def to_ffm(self, matrix, y, path, buffer=10, verbose=True):
        y = np.array(y).astype(int)
        with open(path, "w", buffer) as f:
            for row, label in tqdm.tqdm(zip(matrix, y), disable=(not verbose)):
                f.write(self.format_row(row, label))
                f.write("\n")
