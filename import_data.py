import pandas as pd


def import_data(d=3, model='Xgb'):
    # d is the matrix dimension (=w=h) can be equal [3,5,9], model should be one of ['Xgb','MLP','ResNet']

    if model == 'Xgb':
        if d == 3:
            sig_url = 'https://cernbox.cern.ch/remote.php/dav/public-files/tCQYSTiBoIlrxdV/sig_bdtvars_3by3.csv'
            bg_url = 'https://cernbox.cern.ch/remote.php/dav/public-files/vxzcINZvUxgtzBy/bg_Xgb_vars_3by3.csv'
        if d == 5:
            sig_url = 'https://cernbox.cern.ch/remote.php/dav/public-files/tNBRXxpamrPgZvy/sig_bdtvars_5by5.csv'
            bg_url = 'data/bg_Xgb_vars_' + str(d) + 'by' + str(d) + '.csv'
        if d == 9:
            sig_url = 'data/sig_Xgb_vars_' + str(d) + 'by' + str(d) + '.csv'
            bg_url = 'data/bg_Xgb_vars_' + str(d) + 'by' + str(d) + '.csv'
    else:
        if d == 3:
            sig_url = 'https://cernbox.cern.ch/remote.php/dav/public-files/dQvEYApWwqAUEzJ/signalFile_v2.csv'
            bg_url = 'https://cernbox.cern.ch/remote.php/dav/public-files/cSoWkqPyHBgXvGz/bkgFile_1.csv'
        if d == 5:
            sig_url = 'https://cernbox.cern.ch/remote.php/dav/public-files/tBUrUJzsYleDLUZ/signalFile.csv'
            bg_url = 'https://cernbox.cern.ch/remote.php/dav/public-files/X61ooTzIuVm9ASf/bkgFile.csv'
        if d == 9:
            sig_url = 'data/sig_9by9.csv'
            bg_url = 'data/bg_9by9.csv'

    sig = pd.read_csv(sig_url)
    bg = pd.read_csv(bg_url, nrows=200000)
    sig['signal'] = int(1)
    bg['signal'] = int(0)
    return bg, sig
