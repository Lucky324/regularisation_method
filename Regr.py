import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.keras import models
from tensorflow.keras import layers
# from tensorflow.keras.layers import advanced_activations
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras import optimizers
# from keras import regularizers
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
import os
from tensorflow.keras import initializers as initi
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks as cb
from tensorflow.keras import regularizers
from scipy import stats as st
from tensorflow.keras.models import load_model


# tf.config.experimental.list_physical_devices('GPU')


def coeff_determination(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    SS_res = np.sum(np.square(y_true - y_pred))
    SS_tot = (len(y_true) - 1) * np.var(y_true, ddof=1)
    return (1 - SS_res / (SS_tot))
    # return  np.corrcoef(y_true, y_pred)[0, 1]**2 np.sum(np.square( y_true - np.mean(y_true) ) )


def r_square(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / SS_tot)


class EarlyStoppingAtMinCD(cb.Callback):

    def __init__(self, patience=6000):
        super(EarlyStoppingAtMinCD, self).__init__()

        self.patience = patience

        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
        self.best_weights_next = None
        self.best_epoch = 0

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = -np.Inf
        self.cdL = list(range(MaxEp));
        self.cdT = list(range(MaxEp));
        self.elist = list(range(MaxEp));

    def on_epoch_end(self, epoch, logs=None):
        predictions_test = self.model.predict(test_x)
        current = coeff_determination(test_y.values[:, 0], predictions_test[:, 0])
        predictions_train = self.model.predict(train_x)
        current_train = coeff_determination(train_y.values[:, 0], predictions_train[:, 0])
        self.cdL[epoch] = current_train
        self.cdT[epoch] = current
        #    self.elist[epoch] = epoch
        if epoch == 0:
            predictions_test = self.model.predict(test_x)
            current = coeff_determination(test_y.values[:, 0], predictions_test[:, 0])
            predictions_train = self.model.predict(train_x)
            current_train = coeff_determination(train_y.values[:, 0], predictions_train[:, 0])
            y_pred = self.model.predict(features)
            new_train = train_y.filter(['BP'], axis=1).reset_index(drop=True)
            new_train['BP_pred'] = predictions_train[:, 0]
            new_test = test_y.filter(['BP'], axis=1).reset_index(drop=True)
            new_test['BP_pred'] = predictions_test[:, 0]
            new_train.to_csv(
                r'D:\Projects\Python\Dropout\12_8\BPopt_' + optsStr[opt] + '\№' + str(n1) + '_' + str(n2) + '_' + str(
                    speed) + '\BP_' + str(n1) + '_' + str(n2) + '_' + str(1) + '_' + str(k) + '_reg_' + str(
                    regularize_num) + '_drop_' + str(drop_layer) + '.csv', index=False, sep=';')
            with open(r'D:\Projects\Python\Dropout\12_8\BPopt_' + optsStr[opt] + '\№' + str(n1) + '_' + str(
                    n2) + '_' + str(speed) + '\BP_' + str(n1) + '_' + str(n2) + '_' + str(1) + '_' + str(
                    k) + '_reg_' + str(regularize_num) + '_drop_' + str(drop_layer) + '.csv', 'a') as f:
                new_test.to_csv(f, header=False, index=False, sep=';')
        if epoch in s:
            y_pred = self.model.predict(features)

            new_train = train_y.filter(['BP'], axis=1).reset_index(drop=True)
            new_train['BP_pred'] = predictions_train[:, 0]
            new_test = test_y.filter(['BP'], axis=1).reset_index(drop=True)
            new_test['BP_pred'] = predictions_test[:, 0]

            plt.figure(figsize=(9, 6))
            plt.subplot(1, 1, 1)
            plt.plot(np.arange(20) * 0.05, np.arange(20) * 0.05, 'g')  #
            plt.plot(new_train['BP'], new_train['BP_pred'], 'bo')  # построение графика
            plt.plot(new_test['BP'], new_test['BP_pred'], 'ro')  # построение графика
            plt.title('Эпоха: ' + str(epoch) +
                      '; R^2(train) = ' + str(round(current_train, 4)) +
                      '; R^2(test) = ' + str(round(current, 4)))  # заголовок
            plt.ylabel("Предсказанные значения", fontsize=14)  # ось ординат
            plt.xlabel("Заданные значения", fontsize=14)  # ось ординат
            plt.legend(['Линия регрессии', 'Обучающая (train)', 'Тестовая (test)'], bbox_to_anchor=(1, 1))
            plt.grid(True)  # включение отображение сетки
            plt.savefig(r'D:\Projects\Python\Dropout\12_8\BPopt_' +
                        optsStr[opt] + '\№' + str(n1) + '_' + str(n2) + '_' + str(speed) +
                        '\BP_' + str(n1) + '_' + str(n2) + '_' + str(epoch) + '_' + str(k) + '_reg_' + str(
                regularize_num) + '_drop_' + str(drop_layer) + '.png', format='png')
            # plt.show()

            # fig = go.Figure()
            # fig.add_trace(go.Scatter(
            #               x=np.arange(20) * 0.05,
            #               y=np.arange(20) * 0.05,
            #               line=dict(color="green"),
            #               mode="lines",
            #               name="Линия регрессии",
            #               ))
            # fig.add_trace(go.Scatter(
            #               x=new_train['BP'],
            #               y=new_train['BP_pred'],
            #               marker=dict(color="blue", size=10),
            #               mode="markers",
            #               name="Обучающая (train)",
            #               ))
            # fig.add_trace(go.Scatter(
            #               x=new_test['BP'],
            #               y=new_test['BP_pred'],
            #               marker=dict(color="red", size=10),
            #               mode="markers",
            #               name="Тестовая (test)",
            #               ))
            # fig.update_layout(legend_orientation="h",legend=dict(x=0.1, y=-0.2),
            #                  xaxis=dict(title="Заданные значения",position=0.015),
            #                  yaxis_title="Предсказанные значения",
            #                  annotations=[go.layout.Annotation(
            #                      text='Эпоха: '+str(epoch)+'<br />R<sup>2</sup><sub>train</sub> = '+
            #                           str(round(current_train,4))+'<br />R<sup>2</sup><sub>test</sub> = '+
            #                           str(round(current,4)), align='left', showarrow=False, xref='paper',
            #                            yref='paper', x=0.05, y=0.95, font=dict( color="black", size=14),
            #                            bordercolor='black', borderwidth=1)])
            # fig.write_image(r'D:\Projects\Python\Dropout\12_8\BPopt_' +
            #                optsStr[opt]+'\№' + str(n1)+'_'+str(n2)+'_'+str(speed)+
            #                '\BP_' +  str(n1) + '_' + str(n2) + '_'+ str(epoch)+ '_' + str(k) + '.png')
        if np.greater(current, self.best) and (np.greater(current_train, 0.3) and epoch > 50):
            self.best = current
            self.wait = 0
            self.best_epoch = epoch
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        if np.equal(epoch, self.best_epoch + 1):
            self.best_weights_next = self.model.get_weights()

    #    else:
    #      self.wait += 1
    #      if self.wait >= self.patience:
    #        self.stopped_epoch = epoch
    #        self.model.stop_training = True
    #        print('Restoring model weights from the end of the best epoch %05d.' % (self.best_epoch))
    #        self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        self.ll = {'epoches': self.elist, 'cdL': self.cdL, 'cdT': self.cdT}
        self.df = pd.DataFrame(self.ll)
        self.df.to_csv(
            r'D:\Projects\Python\Dropout\12_8\BPopt_' + optsStr[opt] + '\№' + str(n1) + '_' + str(n2) + '_' + str(
                speed) + '\CD_BP' + '_' + str(n1) + '_' + str(n2) + '_' + str(k) + '_reg_' + str(
                regularize_num) + '_drop_' + str(drop_layer) + '.csv', index=False, sep=';')
        predictions_test = self.model.predict(test_x)
        current = coeff_determination(test_y.values[:, 0], predictions_test[:, 0])
        predictions_train = self.model.predict(train_x)
        current_train = coeff_determination(train_y.values[:, 0], predictions_train[:, 0])
        y_pred = self.model.predict(features)
        new_train = train_y.filter(['BP'], axis=1).reset_index(drop=True)
        new_train['BP_pred'] = predictions_train[:, 0]
        new_test = test_y.filter(['BP'], axis=1).reset_index(drop=True)
        new_test['BP_pred'] = predictions_test[:, 0]

        plt.figure(figsize=(9, 6))
        plt.subplot(1, 1, 1)
        # plt.title(str(n1) + "_" + str(n2) + "_" + str(k) + "; Ошибка и коэф. детерминации")  # заголовок
        plt.plot(np.arange(20) * 0.05, np.arange(20) * 0.05, 'g')  # , name='Линия регрессии')
        plt.plot(new_train['BP'], new_train['BP_pred'], 'bo')  # , name='Обучающая (train)')  # построение графика
        plt.plot(new_test['BP'], new_test['BP_pred'], 'ro')  # , name='Тестовая (test)')  # построение графика
        plt.title('Эпоха: ' + 'end' +
                  '; R^2(train) = ' + str(round(current_train, 4)) +
                  '; R^2(test) = ' + str(round(current, 4)))  # заголовок
        plt.ylabel("Предсказанные значения", fontsize=14)  # ось ординат
        plt.xlabel("Заданные значения", fontsize=14)  # ось ординат
        plt.legend(['Линия регрессии', 'Обучающая (train)', 'Тестовая (test)'], bbox_to_anchor=(1, 1))
        plt.grid(True)  # включение отображение сетки
        plt.savefig(r'D:\Projects\Python\Dropout\12_8\BPopt_' + optsStr[opt] + '\№' +
                    str(n1) + '_' + str(n2) + '_' + str(speed) + '\End_BP_' + str(n1) + '_' + str(n2) + '_' + str(
            k) + '_reg_' + str(regularize_num) + '_drop_' + str(drop_layer) + '.png', format='png')
        # plt.show()

        # fig = go.Figure()
        # fig.add_trace(go.Scatter(
        #                   x=np.arange(20) * 0.05,
        #                   y=np.arange(20) * 0.05,
        #                   line=dict(color="green"),
        #                   mode="lines",
        #                   name="Линия регрессии",
        #                   ))
        # fig.add_trace(go.Scatter(
        #                   x=new_train['BP'],
        #                  y=new_train['BP_pred'],
        #                   marker=dict(color="blue", size=10),
        #                   mode="markers",
        #                   name="Обучающая (train)",
        #                   ))
        # fig.add_trace(go.Scatter(
        #                   x=new_test['BP'],
        #                   y=new_test['BP_pred'],
        #                   marker=dict(color="red", size=10),
        #                   mode="markers",
        #                   name="Тестовая (test)",
        #                   ))
        # fig.update_layout(legend_orientation="h",legend=dict(x=0.1, y=-0.2), xaxis=dict(title="Заданные значения",position=0.015),  yaxis_title="Предсказанные значения",annotations=[go.layout.Annotation(
        #            text='Эпоха: '+'end' + 'R2train='+str(round(current_train,4)) + 'R2test='+str(round(current,4)),
        #            align='left',
        #            showarrow=False,
        #            xref='paper',
        #            yref='paper',
        #            x=0.05,
        #            y=0.95,
        ##            font=dict(
        #            color="black",
        #            size=14
        #            ),
        #            bordercolor='black',
        #            borderwidth=1
        #        )])
        # fig.write_image(r'D:\Projects\Python\Dropout\12_8\BPopt_' + optsStr[opt]+'\№' +
        # str(n1)+'_'+str(n2)+'_'+str(speed)+ '\End_BP_' +  str(n1) + '_' + str(n2) + '_' + str(k) + '.png')
        new_train.to_csv(
            r'D:\Projects\Python\Dropout\12_8\BPopt_' + optsStr[opt] + '\№' + str(n1) + '_' + str(n2) + '_' + str(
                speed) + '\End_BP_' + str(n1) + '_' + str(n2) + '_' + str(k) + '_reg_' + str(
                regularize_num) + '_drop_' + str(drop_layer) + '.csv', index=False, sep=';')
        with open(r'D:\Projects\Python\Dropout\12_8\BPopt_' + optsStr[opt] + '\№' + str(n1) + '_' + str(n2) + '_' + str(
                speed) + '\End_BP_' + str(n1) + '_' + str(n2) + '_' + str(k) + '_reg_' + str(
                regularize_num) + '_drop_' + str(drop_layer) + '.csv', 'a') as f:
            new_test.to_csv(f, header=False, index=False, sep=';')
        if self.best_epoch > 0:
            self.model.set_weights(self.best_weights)
            predictions_test = self.model.predict(test_x)
            current = coeff_determination(test_y.values[:, 0], predictions_test[:, 0])
            predictions_train = self.model.predict(train_x)
            current_train = coeff_determination(train_y.values[:, 0], predictions_train[:, 0])
            y_pred = self.model.predict(features)
            new_train = train_y.filter(['BP'], axis=1).reset_index(drop=True)
            new_train['BP_pred'] = predictions_train[:, 0]
            new_test = test_y.filter(['BP'], axis=1).reset_index(drop=True)
            new_test['BP_pred'] = predictions_test[:, 0]

            plt.figure(figsize=(9, 6))
            plt.subplot(1, 1, 1)
            plt.plot(np.arange(20) * 0.05, np.arange(20) * 0.05, 'g')
            plt.plot(new_train['BP'], new_train['BP_pred'], 'bo')  # построение графика
            plt.plot(new_test['BP'], new_test['BP_pred'], 'ro')  # построение графика
            plt.title('Эпоха: ' + str(self.best_epoch) +
                      '; R^2(train) = ' + str(round(current_train, 4)) +
                      '; R^2(test) = ' + str(round(current, 4)))  # заголовок
            plt.ylabel("Предсказанные значения", fontsize=14)  # ось ординат
            plt.xlabel("Заданные значения", fontsize=14)  # ось ординат
            plt.legend(['Линия регрессии', 'Обучающая (train)', 'Тестовая (test)'], bbox_to_anchor=(1, 1))
            plt.grid(True)  # включение отображение сетки
            plt.savefig(r'D:\Projects\Python\Dropout\12_8\BPopt_' + optsStr[opt] + '\№' + str(n1) + '_' + str(n2) +
                        '_' + str(speed) + '\Best_BP_' + str(n1) + '_' + str(n2) + '_' + str(
                self.best_epoch) + '_' + str(k) + '_reg_' + str(regularize_num) + '_drop_' + str(drop_layer) + '.png',
                        format='png')

            new_train.to_csv(
                r'D:\Projects\Python\Dropout\12_8\BPopt_' + optsStr[opt] + '\№' + str(n1) + '_' + str(n2) + '_' + str(
                    speed) + '\BP_' + str(n1) + '_' + str(n2) + '_' + str(self.best_epoch) + '_' + str(
                    k) + '_reg_' + str(regularize_num) + '_drop_' + str(drop_layer) + '.csv', index=False, sep=';')
            with open(r'D:\Projects\Python\Dropout\12_8\BPopt_' + optsStr[opt] + '\№' + str(n1) + '_' + str(
                    n2) + '_' + str(speed) + '\BP_' + str(n1) + '_' + str(n2) + '_' + str(self.best_epoch) + '_' + str(
                    k) + '_reg_' + str(regularize_num) + '_drop_' + str(drop_layer) + '.csv', 'a') as f:
                new_test.to_csv(f, header=False, index=False, sep=';')

            self.model.set_weights(self.best_weights_next)
            predictions_test = self.model.predict(test_x)
            current = coeff_determination(test_y.values[:, 0], predictions_test[:, 0])
            predictions_train = self.model.predict(train_x)
            current_train = coeff_determination(train_y.values[:, 0], predictions_train[:, 0])
            y_pred = self.model.predict(features)
            new_train = train_y.filter(['BP'], axis=1).reset_index(drop=True)
            new_train['BP_pred'] = predictions_train[:, 0]
            new_test = test_y.filter(['BP'], axis=1).reset_index(drop=True)
            new_test['BP_pred'] = predictions_test[:, 0]

            plt.figure(figsize=(9, 6))
            plt.subplot(1, 1, 1)
            plt.plot(np.arange(20) * 0.05, np.arange(20) * 0.05, 'g')
            plt.plot(new_train['BP'], new_train['BP_pred'], 'bo')  # построение графика
            plt.plot(new_test['BP'], new_test['BP_pred'], 'ro')  # построение графика
            plt.title('Эпоха: ' + str(self.best_epoch + 1) +
                      '; R^2(train) = ' + str(round(current_train, 4)) +
                      '; R^2(test) = ' + str(round(current, 4)))  # заголовок
            plt.ylabel("Предсказанные значения", fontsize=14)  # ось ординат
            plt.xlabel("Заданные значения", fontsize=14)  # ось ординат
            plt.legend(['Линия регрессии', 'Обучающая (train)', 'Тестовая (test)'], bbox_to_anchor=(1, 1))
            plt.grid(True)  # включение отображение сетки
            plt.savefig(r'D:\Projects\Python\Dropout\12_8\BPopt_' + optsStr[opt] + '\№' + str(n1) + '_' + str(n2) + '_' +
                        str(speed) + '\Best_BP_next_' + str(n1) + '_' + str(n2) + '_' + str(
                self.best_epoch) + '_' + str(k) + '_reg_' + str(regularize_num) + '_drop_' + str(drop_layer) + '.png',
                        format='png')

            new_train.to_csv(
                r'D:\Projects\Python\Dropout\12_8\BPopt_' + optsStr[opt] + '\№' + str(n1) + '_' + str(n2) + '_' + str(
                    speed) + '\BP_next_' + str(n1) + '_' + str(n2) + '_' + str(self.best_epoch) + '_' + str(
                    k) + '_reg_' + str(regularize_num) + '_drop_' + str(drop_layer) + '.csv', index=False, sep=';')
            with open(r'D:\Projects\Python\Dropout\12_8\BPopt_' + optsStr[opt] + '\№' + str(n1) + '_' + str(
                    n2) + '_' + str(speed) + '\BP_next_' + str(n1) + '_' + str(n2) + '_' + str(
                    self.best_epoch) + '_' + str(k) + '_reg_' + str(regularize_num) + '_drop_' + str(
                    drop_layer) + '.csv', 'a') as f:
                new_test.to_csv(f, header=False, index=False, sep=';')


# s = set([0,1,25,50,100,250, 500, 750, 1000,1250,1500,  2000, 2500, 3000, 3500,4000,4500,5000,  5500, 6000, 6500, 7000,7500,8000,8500,9000,9500,10000])
s = set([0, 25])
filename = ("otd.csv")
names = ['Mounth', 'T', 'Salinity', 'рН', 'O2', 'H2O_O', 'Chem_O2', 'N', 'NN', 'NHN', 'F', 'BP', 'BE']
dataset = pd.read_csv(filename, sep=';', names=names)
dataset = dataset.drop(0)  # убираем заголовки
# dataset=dataset.drop([9,22,35]) #BE - 7,20
output = dataset[['BP']].astype(float)
features = dataset.drop(['Mounth', 'BP', 'BE'], axis=1).astype(float)  # убираем лишнее, единый формат
# features=(features-features.min())/(features.max()-features.min())
# features=(features-features.mean())/features.std() iloc.drop(features.index[[0,1,2,3, 17,18,19,20, 35,36,37,38]]) initi.random_uniform(minval=0.0001, maxval=0.1)
# output=(output-output.min())/(output.max()-output.min())
# output['BP2']=output
# test_x = features[24:]
# test_y = output[24:]
##test_y['BP2'] = output[24:]
# train_x = features[:24]
# train_y = output[:24]

test_x = features.iloc[13:26]
test_y = output.iloc[13:26]
train_x = features.iloc[list(range(0, 13)) + list(range(26, 39))]
train_y = output.iloc[list(range(0, 13)) + list(range(26, 39))]

# train_y['BP2'] = output[:24]
speed = 0.0005  # 0.00005,0.0001, 0.0005
neur = list(range(12, 13))
neur2 = list(range(8, 9))
n1 = 12
n2 = 8
# gl0=0.65*np.random.randn(10,10)
# gl00=np.random.random_sample((10,))-1.5
# gl1=0.65*np.random.randn(10,n1)
# gl11=np.random.random_sample((n1,))-1.5
# gl2= 0.65*np.random.randn(n1,n2)
# gl22=np.random.random_sample((n2,))-1.5
#            while True:
#               w = 0.65*np.random.randn(n1,1)
#               sta,p = st.shapiro(w)
#               if p>0.7:
#                   break
# gl3=0.65*np.random.randn(n2,1)
# gl33=np.array([0.35])

optsStr = ['SGD', 'Nadam', 'RMSprop']
regularize_arr = [0.1, 0.01, 0.001, 0.0001, 0.00001]
regularize_arr2 = [0.1, 0.01, 0.001, 0.0001, 0.00001]
an_models = [17, 28, 31]
nice_models1 = [1, 7, 9, 13, 16, 18, 19]
nice_models2 = [20, 24, 28, 29, 32, 38, 46]
nice_models12 = [1, 7, 9, 13, 16, 18, 19, 20, 24, 28, 29, 32, 38, 46]
nice_models3 = [28]
reg_l1_l2_arr = ['l1', 'l2']

for k in range(50):
    # print(str(k) + "qwertyuiolkjhgfdszxcvbnm,.ljhgfd")
    opts = [optimizers.SGD(learning_rate=0.0005 * 100), optimizers.Nadam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999),
            optimizers.RMSprop(learning_rate=0.00005)]
    for i in range(len(neur)):
        n1 = neur[i]
        for j in range(len(neur2)):
            n2 = neur2[j]
            if (n2 <= 2 or n2 > n1):
                continue
            #gl0 = 0.65 * np.random.randn(10, 10)
            #gl00 = np.random.random_sample((10,)) - 1.5
            gl1 = 0.65 * np.random.randn(10, n1)
            gl11 = np.random.random_sample((n1,)) - 1.5
            gl2 = 0.65 * np.random.randn(n1, n2)
            gl22 = np.random.random_sample((n2,)) - 1.5
            gl3 = 0.65 * np.random.randn(n2, 1)
            gl33 = np.array([0.35])
            for regularize_num in range(1, 2):
                # gl1=0.65*np.random.randn(10,n1)
                # gl11=np.random.random_sample((n1,))-1.5
                # gl2= 0.65*np.random.randn(n1,n2)
                # gl22=np.random.random_sample((n2,))-1.5
                #            while True:
                #               w = 0.65*np.random.randn(n1,1)
                #               sta,p = st.shapiro(w)
                #               if p>0.7:
                #                   break
                # gl3=0.65*np.random.randn(n2,1)
                # gl33=np.array([0.35])
                for drop_layer in range(0, 1):
                    #for reg_l1_l2 in range(0, 2):
                        for opt in range(1, 2):
                            if not os.path.exists(r'D:\Projects\Python\Dropout\12_8\BPopt_' + optsStr[opt]):
                                os.mkdir(r'D:\Projects\Python\Dropout\12_8\BPopt_' + optsStr[opt])
                                print("Directory ", r'D:\Projects\Python\Dropout\12_8\BPopt_' + optsStr[opt], " Created ")
                            else:
                                print("Directory ", r'D:\Projects\Python\Dropout\12_8\BPopt_' + optsStr[opt],
                                      " already exists")
                            if not os.path.exists(
                                    r'D:\Projects\Python\Dropout\12_8\BPopt_' + optsStr[opt] + '\№' + str(n1) + '_' + str(
                                            n2) + '_' + str(speed)):
                                os.mkdir(
                                    r'D:\Projects\Python\Dropout\12_8\BPopt_' + optsStr[opt] + '\№' + str(n1) + '_' + str(
                                        n2) + '_' + str(speed))
                                print("Directory ",
                                      r'D:\Projects\Python\Dropout\12_8\BPopt_' + optsStr[opt] + '\№' + str(n1) + '_' + str(
                                          n2) + '_' + str(speed), " Created ")
                            else:
                                print("Directory ",
                                      r'D:\Projects\Python\Dropout\12_8\BPopt_' + optsStr[opt] + '\№' + str(n1) + '_' + str(
                                          n2) + '_' + str(speed), " already exists")
                            #               cd=[]
                            #               cdL=[]
                            #               cdT=[]
                            MaxEp = 10000  # if optsStr[opt] == 'Nadam' or speed not in [0.0001,0.0005,0.001,0.005] else 300000
                            print("Текущая модель: K=" + str(k) + " opt=" + str(opt) + " regularize=" + str(
                                regularize_num) + " dropout=" + str(drop_layer))
                            model = models.Sequential()
                            # Input - Layer

                            if (regularize_num == 0):
                                #model.add(layers.Dense(10, input_dim=10, activation='tanh',
                                #                       kernel_initializer=initi.RandomNormal(mean=0.0, stddev=0.8,
                                #                                                             seed=None),
                                #                       use_bias=True,
                                #                       bias_initializer=initi.RandomNormal(mean=0.0, stddev=0.2,
                                #                                                           seed=None)))
                                model.add(layers.Dense(n1, input_dim=10, activation='tanh',
                                                       kernel_initializer=initi.RandomNormal(mean=0.0, stddev=0.2,
                                                                                             seed=None),
                                                       use_bias=True,
                                                       bias_initializer=initi.RandomNormal(mean=0.0, stddev=0.2,
                                                                                           seed=None)))
                                if (drop_layer == 1):
                                    model.add(layers.Dropout(0.1))
                                model.add(layers.Dense(n2, activation='tanh',
                                                       kernel_initializer=initi.RandomNormal(mean=0.0, stddev=0.2,
                                                                                             seed=None),
                                                       use_bias=True,
                                                       bias_initializer=initi.RandomNormal(mean=0.0, stddev=0.2,
                                                                                           seed=None)))
                                # Output- Layer
                                model.add(layers.Dense(1, activation='elu',
                                                       kernel_initializer=initi.RandomNormal(mean=0.0, stddev=0.2,
                                                                                             seed=None),
                                                       use_bias=True,
                                                       bias_initializer=initi.RandomNormal(mean=0.0, stddev=0.2,
                                                                                           seed=None)))
                            else:
                                #model.add(layers.Dense(10, input_dim=10, activation='tanh',
                                #                       kernel_initializer=initi.RandomNormal(mean=0.0, stddev=0.8,
                                #                                                             seed=None),
                                #                       use_bias=True,
                                #                       bias_initializer=initi.RandomNormal(mean=0.0, stddev=0.2, seed=None),
                                #                       kernel_regularizer=regularizers.l2(
                                #                           regularize_arr[regularize_num - 1]),
                                #                       bias_regularizer=regularizers.l2(
                                #                           regularize_arr[regularize_num - 1])))
                                model.add(layers.Dense(n1, input_dim=10, activation='tanh',
                                                       kernel_initializer=initi.RandomNormal(mean=0.0, stddev=0.2,
                                                                                             seed=None),
                                                       use_bias=True,
                                                       bias_initializer=initi.RandomNormal(mean=0.0, stddev=0.2, seed=None),
                                                       kernel_regularizer=regularizers.l1(0.0001),
                                                       bias_regularizer=regularizers.l1(0.0001)
                                                       ))
                                if (drop_layer == 1):
                                    model.add(layers.Dropout(0.1))
                                model.add(layers.Dense(n2, activation='tanh',
                                                       kernel_initializer=initi.RandomNormal(mean=0.0, stddev=0.2,
                                                                                             seed=None),
                                                       use_bias=True,
                                                       bias_initializer=initi.RandomNormal(mean=0.0, stddev=0.2, seed=None),
                                                       kernel_regularizer=regularizers.l1(0.001),
                                                       bias_regularizer=regularizers.l1(0.001)
                                                       ))
                                # Output- Layer
                                model.add(layers.Dense(1, activation='elu',
                                                       kernel_initializer=initi.RandomNormal(mean=0.0, stddev=0.2,
                                                                                             seed=None),
                                                       use_bias=True,
                                                       bias_initializer=initi.RandomNormal(mean=0.0, stddev=0.2, seed=None),
                                                       kernel_regularizer=regularizers.l2(0.01),
                                                       bias_regularizer=regularizers.l2(0.01)
                                                       ))
                            # if(k == 45):
                            if (drop_layer == 0):
                                #model.layers[0].set_weights([gl0, gl00])
                                model.layers[0].set_weights([gl1, gl11])
                                model.layers[1].set_weights([gl2, gl22])
                                model.layers[2].set_weights([gl3, gl33])
                            #else:
                            #    model.layers[0].set_weights([gl0, gl00])
                            #    model.layers[1].set_weights([gl1, gl11])
                            #    model.layers[3].set_weights([gl2, gl22])
                            #    model.layers[4].set_weights([gl3, gl33])
                            # else:

                            #W = load_model(r'D:\Projects\Python\Dropout\12_8\BPopt_SGD\№12_8_0.0003\initBP_'+ str(n1)+'_'+str(n2)+'_' + str(k)+'_reg_0_drop_0.h5', custom_objects={'r_square': r_square})
                            #model.layers[0].set_weights(W.layers[0].get_weights())
                            #model.layers[1].set_weights(W.layers[1].get_weights())
                            #model.layers[2].set_weights(W.layers[2].get_weights())
                            #model.layers[3].set_weights(W.layers[3].get_weights())
                            #model.add(layers.LeakyRelu2(alpha=0.3))

                            model.compile(loss='mean_squared_error', optimizer=opts[opt], metrics=[r_square])
                            filenameE = r'D:\Projects\Python\Dropout\12_8\BPopt_' + optsStr[opt] + '\№' + str(
                                n1) + '_' + str(n2) + '_' + str(speed) + '\logBP_' + '_' + str(n1) + '_' + str(
                                n2) + '_' + str(k) + '_reg_' + str(regularize_num) + '_drop_' + str(drop_layer) + '.csv'
                            csv_logger = CSVLogger(filenameE, append=True, separator=';')
                            model.save(r'D:\Projects\Python\Dropout\12_8\BPopt_' + optsStr[opt] + '\№' + str(n1) + '_' + str(
                                n2) + '_' + str(speed) + '\initBP' + '_' + str(n1) + '_' + str(n2) + '_' + str(
                                k) + '_reg_' + str(regularize_num) + '_drop_' + str(drop_layer) + '.h5')
                            # test_loss, test_acc = model.evaluate(train_x, train_y, verbose=2)
                            # print("//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////")
                            # print("Начальная точность: ", test_acc)
                            # print("//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////")
                            loss = []
                            val_loss = []
                            #                for i in range(len(epoch)):
                            time.sleep(1)
                            results = model.fit(
                                train_x, train_y,
                                epochs=MaxEp,
                                batch_size=26,
                                validation_data=(test_x, test_y),
                                callbacks=[csv_logger, EarlyStoppingAtMinCD()]
                            )
                            # val_loss.extend(results.history['val_loss'])
                            n = range(len(loss))
                            predictions_train = model.predict(train_x)
                            predictions_test = model.predict(test_x)
                            y_pred = model.predict(features)
                            new_train = train_y.filter(['BP'], axis=1).reset_index(drop=True)
                            new_train['BP_pred'] = predictions_train[:, 0]
                            #               new_train['BP_pred2']= predictions_train[:,1]
                            new_test = test_y.filter(['BP'], axis=1).reset_index(drop=True)
                            new_test['BP_pred'] = predictions_test[:, 0]
                            #               new_test['BP_pred2']= predictions_test[:,1]

                            model.save(r'D:\Projects\Python\Dropout\12_8\BPopt_' + optsStr[opt] + '\№' + str(n1) + '_' + str(
                                n2) + '_' + str(speed) + '\BP' + '_' + str(
                                n1) + '_' + str(n2) + '_' + str(k) + '_reg_' + str(regularize_num) + '_drop_' + str(
                                drop_layer) + '.h5')
                            plt.clf()
                            datasetE = pd.read_csv(filenameE, sep=';',
                                                   names=['epoch', 'loss', 'r_square', 'val_loss', 'val_r_square'])
                            datasetE = datasetE.drop(0).astype(float)
                            #                errors = datasetE.drop(['epoch', 'r_square', 'val_r_square'], axis=1).astype(float)
                            datasetCD = pd.read_csv(
                                r'D:\Projects\Python\Dropout\12_8\BPopt_' + optsStr[opt] + '\№' + str(n1) + '_' + str(
                                    n2) + '_' + str(speed) + '\CD_BP' + '_' + str(n1) + '_' + str(n2) + '_' + str(
                                    k) + '_reg_' + str(regularize_num) + '_drop_' + str(drop_layer) + '.csv', sep=';',
                                names=['epoches', 'cdL', 'cdT'])
                            datasetCD = datasetCD.drop([0]).astype(float)

                            plt.figure(figsize=(9, 9))
                            plt.subplot(2, 1, 1)
                            axes = plt.gca()
                            axes.set_ylim([0, 1.5])
                            plt.plot(datasetE['epoch'], datasetE['loss'], 'b')  # построение графика
                            plt.plot(datasetE['epoch'], datasetE['val_loss'], 'r')  # построение графика
                            plt.title(str(n1) + "_" + str(n2) + "_" + str(k) + "; Ошибка и коэф. детерминации")  # заголовок
                            plt.ylabel("Значения ошибки", fontsize=14)  # ось ординат
                            plt.grid(True)  # включение отображение сетки
                            plt.subplot(2, 1, 2)
                            axes = plt.gca()
                            axes.set_ylim([-0.5, 1])
                            plt.plot(datasetCD['epoches'], datasetCD['cdL'], 'b')  # построение графика
                            plt.plot(datasetCD['epoches'], datasetCD['cdT'], 'r')  # построение графика
                            plt.ylabel("Значения коэффициента", fontsize=14)  # ось ординат
                            plt.legend(['Обучающая (train)', 'Тестовая (test)'])
                            plt.grid(True)  # включение отображение сетки
                            plt.savefig(r'D:\Projects\Python\Dropout\12_8\BPopt_' + optsStr[opt] + '\№' +
                                            str(n1) + '_' + str(n2) + '_' + str(speed) + '\logBP' + '_' + str(n1) + '_' +
                                            str(n2) + '_' + str(k) + '_reg_' + str(regularize_num) + '_drop_' + str(
                                    drop_layer) + '.png', format='png')

                            path = r'D:\Projects\Python\Dropout\12_8\BPopt_' + optsStr[opt] + '\№'
                            folder = str(n1) + '_' + str(n2)