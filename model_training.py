import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, BatchNormalization, \
    Activation, GlobalAveragePooling2D
from keras.models import Sequential
from keras.utils.np_utils import to_categorical


class CNNTauTriggerDataset():
    def __init__(self, df, d=3):
        X = df.drop(columns=['signal'])
        samples = X.count()[0]
        dfs = np.split(X, [d ** 2], axis=1)
        EM0 = dfs[0]
        EM1 = dfs[1]

        X = np.concatenate([EM0, EM1], axis=1)
        X = np.reshape(X, (samples, 2, d, d))

        self.ys = to_categorical(df['signal'], num_classes=2)
        self.xs = X
        self.n_events = self.ys.size

    def append_data(self, df):
        obj = CNNTauTriggerDataset(df)
        self.xs = np.concatenate([self.xs, obj.xs], axis=0)
        self.ys = np.concatenate([self.ys, obj.ys], axis=0)

        self.n_events += len(obj.ys)

    def __len__(self):
        return self.n_events

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]


class CellBlock(layers.Layer):
    def __init__(self, filter_num, stride=1, d=3):
        super(CellBlock, self).__init__()

        self.conv1 = Conv2D(filter_num[0], (d, d), strides=stride, padding='same')
        self.bn1 = BatchNormalization()
        self.relu1 = Activation('relu')

        self.conv2 = Conv2D(filter_num[1], (d, d), strides=1, padding='same')
        self.bn2 = BatchNormalization()
        self.relu2 = Activation('relu')

        self.residual = Conv2D(filter_num[1], (1, 1), strides=stride, padding='same')

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        r = self.residual(inputs)

        x = layers.add([x, r])
        output = tf.nn.relu(x)

        return output


class ResNet(models.Model):
    def __init__(self, layers_dims, nb_classes=2, d=3):
        super(ResNet, self).__init__()

        self.stem = Sequential([
            Conv2D(16, (d, d), strides=(2, 2), padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D((d, d), strides=(2, 2), padding='same')
        ])  # Start module

        # Number of filters in different convolutional layers
        filter_block1 = [8, 8]

        self.layer1 = self.build_cellblock(filter_block1, layers_dims[0])

        self.avgpool = GlobalAveragePooling2D()
        self.fc = Dense(nb_classes, activation='softmax')

    def call(self, inputs, training=None):
        x = self.stem(inputs)

        x = self.layer1(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x

    def build_cellblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        res_blocks.add(CellBlock(filter_num, stride))  # The first block stride of each layer may be non-1

        for _ in range(1, blocks):
            print(blocks)  # How many blocks each layer consists of
            res_blocks.add(CellBlock(filter_num, stride=1))
        return res_blocks


def build_ResNet(d=3):
    model = ResNet([2], 2)
    model.build(input_shape=(None, 2, d, d))
    return model


def train_model(sig, bg, d=3, model='Xgb', normalize=True):
    df_tot = pd.concat([sig, bg], axis=0)

    if model == 'Xgb':
        model_features = ['reco_pt', 'reco_eta', 'reco_depth', 'reco_density', 'reco_width',
                          'reco_frac0', 'reco_frac1', 'reco_max1', 'reco_max2', 'reco_maxratio',
                          'reco_maxdist', 'reco_maxf1', 'reco_maxf2', 'reco_maxf3',
                          'reco_rmoment2_0', 'reco_rmoment2_1', 'reco_dmoment2']
    else:
        model_features = df_tot.columns[-2 * d ** 2:]

    df_tot = df_tot.dropna(subset=model_features)
    unique_events = df_tot.drop_duplicates(subset=['signal', 'evtNumber'])
    unique_events = unique_events[['signal', 'evtNumber']]

    events_train, events_test, y_train, y_test = train_test_split(unique_events, unique_events[['signal']],
                                                                  test_size=0.3, random_state=42)
    events_val, events_test, y_val, y_test = train_test_split(events_test, y_test, test_size=0.5, random_state=42)

    df_train = df_tot.merge(events_train.reset_index(), left_on=['signal', 'evtNumber'],
                            right_on=['signal', 'evtNumber'])
    df_val = df_tot.merge(events_val.reset_index(), left_on=['signal', 'evtNumber'], right_on=['signal', 'evtNumber'])
    df_test = df_tot.merge(events_test.reset_index(), left_on=['signal', 'evtNumber'], right_on=['signal', 'evtNumber'])

    X_train, y_train = df_train[model_features], df_train['signal']
    X_val, y_val = df_val[model_features], df_val['signal']
    X_test, y_test = df_test[model_features], df_test['signal']

    if model == 'Xgb':
        model = xgb.XGBClassifier(base_score=0.5, objective='binary:logistic', subsample=0.8, colsample_bytree=0.8,
                                  colsample_bylevel=0.8, colsample_bynode=0.8, learning_rate=0.15, max_depth=4,
                                  reg_alpha=10, reg_lambda=10, gamma=10, n_estimators=1000)
        model.fit(df_train[model_features], np.array(y_train), eval_set=[(df_val[model_features], np.array(y_val))],
                  verbose=1, early_stopping_rounds=15)
        df_test['score'] = model.predict_proba(df_test[model_features])[:, 1]
        return df_test

    if normalize == True:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = pd.DataFrame(scaler.transform(X_train))
        X_val = pd.DataFrame(scaler.transform(X_val))
        X_test = pd.DataFrame(scaler.transform(X_test))

    if model == 'MLP':
        model = MLPClassifier(hidden_layer_sizes=(4, 5), solver='adam', early_stopping=True, random_state=5,
                              verbose=True, learning_rate_init=0.01)
        model.fit(X_train, y_train)
        df_test['score'] = model.predict_proba(X_test)[:, 1]

    else:
        dataset_cnn_train = CNNTauTriggerDataset(pd.concat([X_train, y_train], axis=1), d)
        dataset_cnn_val = CNNTauTriggerDataset(pd.concat([X_val, y_val], axis=1), d)
        dataset_cnn_test = CNNTauTriggerDataset(pd.concat([X_test, y_test], axis=1), d)
        model = build_ResNet(d)
        model.compile(loss=tf.losses.categorical_crossentropy, optimizer=tf.optimizers.SGD(learning_rate=0.05),
                      metrics=['accuracy', 'AUC'])  # BinaryCrossEntropy
        model.fit(dataset_cnn_train.xs, dataset_cnn_train.ys, validation_data=(dataset_cnn_val.xs, dataset_cnn_val.ys),
                  batch_size=32, epochs=4)
        df_test['score'] = model.predict(dataset_cnn_test.xs)[:, 1]

    return df_test
