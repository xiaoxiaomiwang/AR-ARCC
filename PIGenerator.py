import os
import sys
import time
import utils
import torch
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt
from models.NNModel import NNModel
from sklearn.model_selection import KFold
from Datasets.GenerateDatasets import DataLoader
from sklearn.model_selection import train_test_split
from models.aggregation_functions import _split_normal_aggregator

class PIGenerator:

    def __init__(self, dataset=None, method=None):

        self.dataset = dataset
        self.method = method
        dataLoader = DataLoader(dataset=dataset)
        self.X, self.Y = dataLoader.X, dataLoader.Y

        self.kfold = KFold(n_splits=10, shuffle=True, random_state=13)

        print("Loading model...")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.reset_model()

    def reset_model(self):
        return NNModel(device=self.device, nfeatures=self.X.shape[1], method=self.method)


    def train(self, crossval=None, batch_size=None, epochs=None, alpha_=None, printProcess=True):

        cvmse, cvpicp, cvmpiw, cvdiffs = [], [], [], []
        ypred, y_u, y_l, iterator = None, None, None, None


        folder = "CVResults//" + self.dataset + "//" + self.method
        if not os.path.exists("CVResults//" + self.dataset):
            os.mkdir("CVResults//" + self.dataset)
        if not os.path.exists(folder):
            os.mkdir(folder)

        if crossval == "10x1":
            iterator = self.kfold.split(self.X)
            print("Using 10x1 cross-validation for this dataset")
        elif crossval == "5x2":

            seeds = [13, 51, 137, 24659, 347, 436, 123, 64, 958, 234]
            iterator = enumerate(seeds)
            print("Using 5x2 cross-validation for this dataset")
        else:
            sys.exit("Only '10x1' and '5x2' cross-validation are permited.")

        ntrain = 1

        for first, second in iterator:
            if ntrain >= 1:
                if crossval == '10x1':

                    train = np.array(first)
                    test = np.array(second)
                else:

                    train, test = train_test_split(range(len(self.X)), test_size=0.50, random_state=second)
                    train = np.array(train)
                    test = np.array(test)

                print("\n******************************")
                print("Training fold: " + str(ntrain))
                print("******************************")

                Xtrain, means, stds = utils.normalize(self.X[train])
                Ytrain, maxs, mins = utils.minMaxScale(self.Y[train])
                Xval = utils.applynormalize(self.X[test], means, stds)
                Yval = utils.applyMinMaxScale(self.Y[test], maxs, mins)

                filepath = folder + "//weights-" + self.method + "-" + self.dataset + "-" + str(ntrain)

                mse, PICP, MPIW = None, None, None
                if self.method in ['AR_ARCC', 'MCDropout']:
                    m = 1
                else:
                    m = 5
                    filepath = [filepath] * m
                for mi in range(m):
                    if self.method in ['AR_ARCC', 'MCDropout']:
                        f = filepath
                    else:
                        filepath[mi] = filepath[mi] + "-Model" + str(mi)
                        f = filepath[mi]
                    self.model = self.reset_model()
                    _, _, _, mse, PICP, MPIW = self.model.trainFold(Xtrain=Xtrain, Ytrain=Ytrain, Xval=Xval, Yval=Yval,
                                                                    batch_size=batch_size, epochs=epochs, filepath=f,
                                                                    printProcess=printProcess, alpha_=alpha_,
                                                                    yscale=[maxs, mins])
                if self.method not in ['AR_ARCC']:
                    [mse, PICP, MPIW, ypred, y_u, y_l] = self.calculate_metrics(Xval, Yval, maxs, mins, filepath)
                print('PERFORMANCE AFTER AGGREGATION:')
                print("Val MSE: " + str(mse) + " Val PICP: " + str(PICP) + " Val MPIW: " + str(MPIW))

                cvmse.append(mse)
                cvpicp.append(PICP)
                cvmpiw.append(MPIW)


                if self.dataset == "Synth":
                    if self.method in ['AR_ARCC']:
                        self.model.loadModel(filepath)
                        yout = self.model.evaluateFoldUncertainty(valxn=Xval, maxs=None, mins=None, batch_size=32,
                                                                  MC_samples=50)
                        yout = np.array(yout)

                        y_u = np.mean(yout[:, 0], axis=1)
                        y_l = np.mean(yout[:, 1], axis=1)
                        ypred = np.mean(yout[:, 2], axis=1)
                        ypred = utils.reverseMinMaxScale(ypred, maxs, mins)
                        y_u = utils.reverseMinMaxScale(y_u, maxs, mins)
                        y_l = utils.reverseMinMaxScale(y_l, maxs, mins)
                    Xvalp = utils.reversenormalize(Xval, means, stds)
                    _, _, P1, P2 = utils.create_synth_data(plot=True)
                    diffs = 0
                    for iv, x in enumerate(test):
                        ubound, lbound = P1[x], P2[x]
                        diffs += np.abs(ubound - y_u[iv]) + np.abs(y_l[iv] - lbound)
                    cvdiffs.append(diffs)
                    plt.scatter(Xvalp[:, 0], ypred, label='Predicted Data', s=24)
                    plt.scatter(Xvalp[:, 0], y_u, label='Predicted Upper Bounds', s=24)
                    plt.scatter(Xvalp[:, 0], y_l, label='Predicted Lower Bounds', s=24, c='gold')
                    plt.legend(bbox_to_anchor=(1.06, 0.6), fontsize=18)
                    plt.title(self.method, fontsize=24)
                    plt.xlabel('x', fontsize=22)
                    plt.ylabel('y', fontsize=22)
                    plt.xticks(fontsize=22)
                    plt.yticks(fontsize=22)


                self.model = self.reset_model()

            ntrain += 1

        np.save(folder + '//validation_MSE-' + self.method + "-" + self.dataset, cvmse)
        np.save(folder + '//validation_MPIW-' + self.method + "-" + self.dataset, cvmpiw)
        np.save(folder + '//validation_PICP-' + self.method + "-" + self.dataset, cvpicp)
        if self.dataset == "Synth":
            np.save(folder + '//validation_DIFFS-' + self.method + "-" + self.dataset, cvdiffs)

        file_name = folder + "//regression_report-" + self.method + "-" + self.dataset + ".txt"
        with open(file_name, 'w') as x_file:
            x_file.write("Overall MSE %.6f (+/- %.6f)" % (float(np.mean(cvmse)), float(np.std(cvmse))))
            x_file.write('\n')
            x_file.write("Overall PICP %.6f (+/- %.6f)" % (float(np.mean(cvpicp)), float(np.std(cvpicp))))
            x_file.write('\n')
            x_file.write("Overall MPIW %.6f (+/- %.6f)" % (float(np.mean(cvmpiw)), float(np.std(cvmpiw))))
            if self.dataset == "Synth":
                x_file.write('\n')
                x_file.write("Overall DIFF %.6f (+/- %.6f)" % (float(np.mean(cvdiffs)), float(np.std(cvdiffs))))
        return cvmse, cvmpiw, cvpicp

    def calculate_metrics(self, Xval, Yval, maxs, mins, filepath=None):

        startsplit = time.time()

        if self.method in ['AR_ARCC', 'MCDropout']:
            self.model.loadModel(filepath)  # Load model

            yout = self.model.evaluateFoldUncertainty(valxn=Xval, maxs=None, mins=None, batch_size=32, MC_samples=50)
            yout = np.array(yout)
            if self.method in ['AQD', 'AR_ARCC']:

                y_u = np.mean(yout[:, 0], axis=1)
                y_l = np.mean(yout[:, 1], axis=1)

                ypred = np.mean(yout[:, 2], axis=1)
                ypred = utils.reverseMinMaxScale(ypred, maxs, mins)
                y_u = utils.reverseMinMaxScale(y_u, maxs, mins)
                y_l = utils.reverseMinMaxScale(y_l, maxs, mins)
            else:

                with open(filepath + '_validationMSE', 'rb') as f:
                    val_MSE = pickle.load(f)

                yout = utils.reverseMinMaxScale(yout, maxs, mins)
                ypred = np.mean(yout, axis=1)

                model_uncertainty = np.std(yout, axis=1)
                y_u = ypred + 1.96 * np.sqrt(model_uncertainty ** 2 + val_MSE)
                y_l = ypred - 1.96 * np.sqrt(model_uncertainty ** 2 + val_MSE)
        else:
            yout = np.zeros((len(Xval), 3, len(filepath)))
            y_l = np.zeros((len(Xval)))
            y_u = np.zeros((len(Xval)))
            ypred = np.zeros((len(Xval)))

            for mi in range(len(filepath)):
                self.model.loadModel(filepath[mi])

                yout[:, :, mi] = self.model.evaluateFoldUncertainty(valxn=Xval,
                                                                    maxs=None, mins=None, batch_size=5000,
                                                                    MC_samples=1)[:, :, 0]
            if self.method == 'QD':

                y_u = np.mean(yout[:, 0], axis=1) + 1.96 * np.std(yout[:, 0], axis=1)
                y_l = np.mean(yout[:, 1], axis=1) - 1.96 * np.std(yout[:, 1], axis=1)

                ypred = np.mean(yout[:, 2], axis=1)
            else:

                for s in range(len(yout)):
                    yp = yout[s, :, :].transpose()

                    yp[:, [1, 0]] = yp[:, [0, 1]]
                    y_p_agg, y_l_agg, y_u_agg = _split_normal_aggregator(alpha=0.05, y_pred=yp, seed=7)
                    y_l[s] = y_l_agg
                    y_u[s] = y_u_agg
                    ypred[s] = y_p_agg

            ypred = utils.reverseMinMaxScale(ypred, maxs, mins)
            y_u = utils.reverseMinMaxScale(y_u, maxs, mins)
            y_l = utils.reverseMinMaxScale(y_l, maxs, mins)


        Yval = utils.reverseMinMaxScale(Yval, maxs, mins)


        mse = utils.mse(Yval, ypred)

        y_true = torch.from_numpy(Yval).float().to(self.device)
        y_ut = torch.from_numpy(y_u).float().to(self.device)
        y_lt = torch.from_numpy(y_l).float().to(self.device)
        K_U = torch.max(torch.zeros(y_true.size()).to(self.device), torch.sign(y_ut - y_true))
        K_L = torch.max(torch.zeros(y_true.size()).to(self.device), torch.sign(y_true - y_lt))
        K = torch.mul(K_U, K_L)

        MPIW = torch.mean(y_ut - y_lt).item()

        PICP = torch.mean(K).item()

        endsplit = time.time()
        print("It took " + str(endsplit - startsplit) + " seconds to execute this batch")

        return [mse, PICP, MPIW, ypred, y_u, y_l]

    def tune(self):

        folder = "TuningResults//" + self.dataset + "//" + self.method
        if not os.path.exists("TuningResults//" + self.dataset):
            os.mkdir("TuningResults//" + self.dataset)
        if not os.path.exists(folder):
            os.mkdir(folder)

        epochs = 2000
        if self.dataset == 'Kin8nm':
            epochs = 400
        elif self.dataset == 'Naval':
            epochs = 400
        elif self.dataset == 'Boston':
            epochs = 3500
        elif self.dataset == 'Concrete':
            epochs = 2500
        elif self.dataset == 'Energy':
            epochs = 1500
        elif self.dataset == 'Yacht':
            epochs = 4500
        elif self.dataset == 'Wine':
            epochs = 500
        elif self.dataset == 'Protein':
            epochs = 3000
        elif self.dataset == 'Power':
            epochs = 4000

        if self.method == 'AR_ARCC':
            beta_ = [0.001, 0.005, 0.01, 0.05, 0.1]  # [0.01, 0.05, 0.1]
        elif self.method == 'QD+':
            lambda_1 = np.arange(0.2, 1, .1)
            lambda_2 = np.arange(0.2, .6, .1)

            beta_ = list(itertools.product(lambda_1, lambda_2))
        else:
            beta_ = np.arange(0.021054, 0.05, 0.0025)

        count = 0
        results = []
        for bi in beta_:
            print("*****************************************")
            print("Trainining: " + str(count) + " / " + str(len(beta_)))
            print("*****************************************")
            iterator = self.kfold.split(self.X)
            count += 1
            ntrain = 1
            cvmse = []
            cvpicp = []
            cvmpiw = []
            for first, second in iterator:
                if ntrain >= 1:
                    train = np.array(first)
                    test = np.array(second)
                    print("\n******************************")
                    print("Starting fold: " + str(ntrain))
                    print("******************************")

                    filepath = folder + "//weights-" + self.method + "-" + self.dataset + "-temp"
                    if self.method == 'AR_ARCC':
                        filepath = folder + "//weights-" + self.method + "-" + self.dataset + "-" + str(ntrain)

                    Xtrain, means, stds = utils.normalize(self.X[train])
                    Ytrain, maxs, mins = utils.minMaxScale(self.Y[train])
                    Xval = utils.applynormalize(self.X[test], means, stds)
                    Yval = utils.applyMinMaxScale(self.Y[test], maxs, mins)
                    metrics = None

                    if self.method in ['AQD', 'AR_ARCC', 'MCDropout']:
                        m = 1
                    else:
                        m = 5

                        filepath = [filepath] * m
                    for mi in range(m):
                        if self.method in ['AQD', 'AR_ARCC', 'MCDropout']:
                            f = filepath
                        else:
                            filepath[mi] = filepath[mi] + "-Model" + str(mi)
                            f = filepath[mi]
                        metrics = self.model.trainFold(Xtrain=Xtrain, Ytrain=Ytrain, Xval=Xval, Yval=Yval,
                                                       batch_size=16, epochs=epochs, filepath=f, printProcess=True,
                                                       alpha_=bi, yscale=[maxs, mins])

                        self.model = self.reset_model()


                    if self.method not in ['AQD',
                                           'AR_ARCC']:
                        metrics = self.calculate_metrics(Xval, Yval, maxs, mins, filepath)
                    cvmse.append(metrics[3])
                    cvpicp.append(metrics[4])
                    cvmpiw.append(metrics[5])
                    print("############################")
                    print('FOLD PERFORMANCE')
                    print("############################")
                    print("Val MSE: " + str(metrics[3]) + " Val PICP: " + str(metrics[4]) + " Val MPIW: " + str(metrics[5]))

                    self.model = self.reset_model()
                ntrain += 1

            print("########################################")
            print('AVERAGE CV PERFORMANCE AFTER AGGREGATION')
            print("########################################")
            av_mse, av_picp, av_mpiw = np.mean(cvmse), np.mean(cvpicp), np.mean(cvmpiw)
            print("Val MSE: " + str(av_mse) + " Val PICP: " + str(av_picp) +
                  " Val MPIW: " + str(np.mean(av_mpiw)))
            results.append([np.mean(cvmse), np.mean(cvpicp), np.mean(cvmpiw)])

            file_name = folder + "//tuning_results.txt"
            if self.method in ['AR_ARCC', 'QD']:
                with open(file_name, 'a') as x_file:
                    x_file.write("Beta %.6f%%: MSE %.6f%%, PICP %.6f%%, MPIW %.6f%%" %
                                 (float(bi), float(av_mse), float(av_picp), float(av_mpiw)))
                    x_file.write('\n')
            else:
                with open(file_name, 'a') as x_file:
                    x_file.write("Lambda_1 %.6f%% - Lambda_2 %.6f%%: MSE %.6f%%, PICP %.6f%%, MPIW %.6f%%" %
                                 (float(bi[0]), float(bi[1]), float(av_mse), float(av_picp), float(av_mpiw)))
                    x_file.write('\n')

        np.save(folder + '//tuning_results_' + self.method + '-' + self.dataset + '.npy', np.array(results))

if __name__ == '__main__':


    name = 'Boston'
    predictor = PIGenerator(dataset=name, method='AR_ARCC')
    cvmse, cvmpiw, cvpicp = predictor.train(crossval='10x1', batch_size=16, epochs=3000, printProcess=False, alpha_=0.005)

    name = 'Concrete'
    predictor = PIGenerator(dataset=name, method='AR_ARCC')
    cvmse, cvmpiw, cvpicp =predictor.train(crossval='10x1', batch_size=16, epochs=3000, printProcess=False, alpha_=0.008)

    name = 'Energy'
    predictor = PIGenerator(dataset=name, method='AR_ARCC')
    cvmse, cvmpiw, cvpicp = predictor.train(crossval='10x1', batch_size=16, epochs=3500, printProcess=False, alpha_=1)

    name = 'Kin8nm'
    predictor = PIGenerator(dataset=name, method='AR_ARCC')
    cvmse, cvmpiw, cvpicp =predictor.train(crossval='10x1', batch_size=16, epochs=1000, printProcess=False, alpha_=0.0008)
    #######################################################################################################################
    name = 'Power'
    predictor = PIGenerator(dataset=name, method='AR_ARCC')
    cvmse, cvmpiw, cvpicp = predictor.train(crossval='10x1', batch_size=16, epochs=4000, printProcess=False, alpha_=0.05)
    10.03
    name = 'Protein'
    predictor = PIGenerator(dataset=name, method='AR_ARCC')
    cvmse, cvmpiw, cvpicp =predictor.train(crossval='10x1', batch_size=512, epochs=3500, printProcess=False, alpha_=0.5)
    name = 'Year'
    predictor = PIGenerator(dataset=name, method='AR_ARCC')
    cvmse, cvmpiw, cvpicp =predictor.train(crossval='10x1', batch_size=512, epochs=200, printProcess=False, alpha_=0.008)
    name = 'Yacht'
    predictor = PIGenerator(dataset=name, method='AR_ARCC')
    cvmse, cvmpiw, cvpicp =predictor.train(crossval='10x1', batch_size=16, epochs=4500, printProcess=True, alpha_=0.004)
    def calculate_std_deviation(numbers):

        numbers = torch.tensor(numbers, dtype=torch.float32)
        numbers_tensor = numbers.clone().detach()

        mean = torch.mean(numbers_tensor)

        variance = torch.mean((numbers_tensor - mean) ** 2)

        std_deviation = torch.sqrt(variance)
        return std_deviation
    cvmse = torch.tensor(cvmse, dtype=torch.float32)
    cvmpiw = torch.tensor(cvmpiw, dtype=torch.float32)
    cvpicp = torch.tensor(cvpicp, dtype=torch.float32)
    std_cvmse = calculate_std_deviation(cvmse)
    std_cvmpiw = calculate_std_deviation(cvmpiw)
    std_cvpicp = calculate_std_deviation(cvpicp)
    print("Val  std_MSE: " + str(std_cvmse) + " Val  std_PICP: " + str(std_cvpicp) + " Val  std_MPIW: " + str(std_cvmpiw))
    print("Val  mean_MSE: " + str(torch.mean(cvmse)) + " Val  mean_PICP: " + str(torch.mean(cvpicp)) + " Val  mean_MPIW: " + str(torch.mean(cvmpiw)))

