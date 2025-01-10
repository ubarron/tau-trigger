import math
import numpy as np


def get_cell_mapping(d, w, h):
    cells = []
    for depth in range(d):
        layer_cells = []
        for eta in range(w):
            phi_cells = []
            for phi in range(h):
                phi_cells += ['C' + str(h * w * depth + w * eta + phi)]
            layer_cells += [phi_cells]
        cells += [layer_cells]
    return cells


def calc_tob_pt(row, shapes, d, w, h):
    cells = get_cell_mapping(d, w, h)

    E = []
    for n in range(len(shapes)):
        E_shape = 0
        for depth in range(d):
            for eta in range(w):
                for phi in range(h):
                    E_shape += row[cells[depth][eta][phi]] * shapes[n][depth][eta][phi]
        E += [E_shape]

    return max(E)


def calc_toc(sig, bg, vis_pt_col, tob_pt_col, bins, rate):
    # find threshold for given rate
    max_evt_pt = bg.groupby(['evtNumber']).max().sort_values(by=[tob_pt_col])
    pt_thresh = max_evt_pt.iloc[int(rate * len(max_evt_pt))][tob_pt_col]

    # get curve
    sig['passed'] = sig.apply(lambda x: x[tob_pt_col] > pt_thresh, axis=1)
    sig['bin'] = sig.apply(lambda x: np.digitize(x[vis_pt_col], bins), axis=1)
    mse = sig.groupby(['bin']).std()['passed'] / sig.groupby(['bin']).count()['passed']
    rmse = np.sqrt(mse)
    return sig.groupby(['bin']).mean()['passed'], rmse


def get_dwh_collection(d, w, h):
    shape_collection = []
    square_d = 3
    for i in range(w - square_d + 1):
        for j in range(h - square_d + 1):
            shp = np.zeros((w, h))
            shp[i:i + square_d, j:j + square_d] = np.ones((square_d, square_d))
            shape_collection += [[shp.tolist()] * 2]

    return [shape_collection]


class TOB:
    def __init__(self, csv_line, layers, rows, cols, col_names):
        tob_info = dict(zip(col_names, csv_line))
        self._layers = layers
        self._rows = rows
        self._cols = cols

        self._pt = float(tob_info['tob_et_em']) + float(tob_info['tob_et_had'])
        self._eta = float(tob_info['tob_eta'])
        self._phi = float(tob_info['tob_phi'])

        num_cells = self._layers * self._rows * self._cols
        cells = np.array([tob_info['C' + str(i)] for i in range(num_cells)], dtype='float64')
        self._cells = cells.reshape(self._layers, self._rows, self._cols).transpose((0, 2, 1))

    def largeTauClus(self):
        return self._pt

    def eta(self):
        return self._eta

    def phi(self):
        return self._phi

    def getEnergy(self, l, r, c):
        return self._cells[l, r, c]

    def clusDepth(self):
        layerDepth = [5, 10]

        myDepth = 0
        for layer in range(self._layers):
            myDepth += layerDepth[layer] * self._cells[layer].sum()

        if self.largeTauClus() < 0.001:
            return 0
        else:
            return myDepth / self.largeTauClus()

    def clusDens(self):
        layerVolume = [147.0 * 147.0 * 10.0, 147.0 * 147.0 * 10.0]

        myDens = 0
        for layer in range(self._layers):
            for cell in self._cells[layer].flatten():
                myDens += (cell * cell) / layerVolume[layer]

        if self.largeTauClus() < 0.001:
            return 0
        else:
            return math.log(myDens / self.largeTauClus())

    def clusWidth(self):
        layerEta = [0.1, 0.1]

        myWidth = 0
        for layer in range(self._layers):
            eta = -0.1
            phi = -0.1
            for cell in self._cells[layer].flatten():
                myWidth += math.sqrt((eta * eta) + (phi * phi)) * cell
                eta += layerEta[layer]
                if eta > 0.101:
                    phi += 0.1
                    eta = -0.1

        if self.largeTauClus() < 0.001:
            return 0
        else:
            return myWidth / self.largeTauClus()

    def clusRmoment2(self, layer):

        layer_energy = self._cells[layer].sum()
        if layer_energy == 0:
            return -1

        layerEta = [0.1, 0.1]
        layerOffset = [1, 1]
        layerCells = [i * 3 for i in layerOffset]

        eta_CoM = 0
        phi_CoM = 0

        eta = np.linspace(-(layerCells[layer] - 1) / 2 * layerEta[layer], (layerCells[layer] - 1) / 2 * layerEta[layer],
                          layerCells[layer])
        phi = np.linspace(-0.1, 0.1, 3)
        count = 0

        for i in range(len(phi)):
            for j in range(len(eta)):
                eta_CoM += self._cells[layer][i][j] * eta[j] / layer_energy
                phi_CoM += self._cells[layer][i][j] * phi[i] / layer_energy
                count += 1

        rmoment2 = 0
        count = 0

        for i in range(len(phi)):
            for j in range(len(eta)):
                rmoment2 += self._cells[layer][i][j] * (
                            (eta[j] - eta_CoM) ** 2 + (phi[i] - phi_CoM) ** 2) / layer_energy
                count += 1

        return rmoment2

    def clusDmoment2(self):
        layerDepth = [5, 10]

        myDmoment2 = 0
        for layer in range(self._layers):
            myDmoment2 += self._cells[layer].sum() * (layerDepth[layer] - self.clusDepth()) ** 2

        if self.largeTauClus() < 0.001:
            return 0
        else:
            return myDmoment2 / self.largeTauClus()

    def layerFrac(self, layer):
        if self.largeTauClus() < 0.001:
            return 0
        else:
            return self._cells[layer].sum() / self.largeTauClus()

    def nearbyCells(self, layer, index1, index2):
        layerCells = [3, 3]

        phiIndex = [0, 0]
        etaIndex = [index1, index2]

        for i in range(2):
            if etaIndex[i] > (layerCells[layer] - 1):
                phiIndex[i] += 1
                etaIndex[i] -= layerCells[layer]
            if etaIndex[i] > (layerCells[layer] - 1):
                phiIndex[i] += 1
                etaIndex[i] -= layerCells[layer]

        distance1 = abs(phiIndex[0] - phiIndex[1]) + abs(etaIndex[0] - etaIndex[1])
        distance2 = 0.1 * abs(phiIndex[0] - phiIndex[1]) + (0.3 / layerCells[layer]) * abs(etaIndex[0] - etaIndex[1])
        return [distance1, distance2]

    def findPeak(self, layer):
        # look for a 1st maximum
        max1 = -1.0
        max1index = -1
        layer_cells = self._cells[layer].flatten()
        for i in range(len(layer_cells)):
            if layer_cells[i] > max1:
                max1 = layer_cells[i]
                max1index = i

        max2 = -1
        max2index = -1

        # find the 2nd maximum, ignore the first (and nearby cells)
        for j in range(len(layer_cells)):
            if layer_cells[j] > max2 and self.nearbyCells(layer, max1index, j)[0] > 1.01:
                max2 = layer_cells[j]
                max2index = j

        ratio = 0
        if max1 > 0.001:
            ratio = max2 / max1

        distance = 0
        if max1index != -1 and max2index != -1:
            distance = self.nearbyCells(layer, max1index, max2index)[1]

        return [max1, max2, ratio, distance, layer_cells.sum(), max1index, max2index]
