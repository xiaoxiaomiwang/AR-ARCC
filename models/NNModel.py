import sys
import utils
import torch
import pickle
import random
import numpy as np
from tqdm import trange
from torch import optim
from models.network import *

# import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
np.random.seed(7)  # Initialize seed to get reproducible results
random.seed(7)
torch.manual_seed(7)
torch.cuda.manual_seed(7)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


#######################################################################################################################
# Static functions and Loss functions 静态函数和损失函数
#######################################################################################################################
def enable_dropout(model):
    """ Function to enable the dropout layers during test-time."""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def AR_ARCC_objective(y_pred, y_true, beta_, pe):
    """Proposed AR_ARCC loss function,
    @param y_pred: NN output (y_u, y_l)
    @param y_true: Ground-truth.
    @param pe: Point estimate (from the base model).
    @param beta_: Specify the importance of the width factor."""
    # Separate upper and lower limits
    y_u = y_pred[:, 0]
    y_l = y_pred[:, 1]
    y_o = pe.detach().squeeze(1)
    if beta_ is not None:

        diff = torch.abs(y_o - y_true).detach()
        cs_greater = torch.max(diff[y_o > y_true]) if y_o[y_o > y_true].numel() > 0 else torch.zeros_like(y_true[0])
        cs_lessequal = torch.max(diff[y_o <= y_true]) if y_o[y_o <= y_true].numel() > 0 else torch.zeros_like(y_true[0])
        mse_loss = nn.MSELoss()(y_true, (y_l + y_u) / 2.0)
        coverage_loss = torch.mean(torch.abs(torch.sign(y_true - y_u) - torch.sign(y_true - y_l)))  # 区间覆盖率损失

        huber_delta = 0.8

        huber_loss_l = torch.where(torch.abs(y_true - y_l) < huber_delta, 0.5 * (y_true - y_l) ** 2,huber_delta * torch.abs(y_true - y_l) - 0.5 * huber_delta ** 2)
        huber_loss_u = torch.where(torch.abs(y_true - y_u) < huber_delta, 0.5 * (y_true - y_u) ** 2, huber_delta * torch.abs(y_true - y_u) - 0.5 * huber_delta ** 2)
        huber_loss = torch.mean(huber_loss_l + huber_loss_u)

        Constraints = torch.exp(torch.mean(-y_u + y_true) + cs_lessequal) + torch.exp(torch.mean(-y_true + y_l) + cs_greater)


        Loss_S = mse_loss * coverage_loss + Constraints * beta_ + huber_loss

    else:

        MPIW_p = torch.mean(torch.abs(y_u - y_o) + torch.abs(y_o - y_l))
        Loss_S = MPIW_p

    return Loss_S


def AQD_objective(y_pred, y_true, beta_):
    """Proposed AQD loss function,
    @param y_pred: NN output (y_u, y_l, y)
    @param y_true: Ground-truth.
    @param beta_: Array of hyperparameters: [lambda_1, lambda_2]."""
    # Separate upper and lower limits
    y_u = y_pred[:, 0]
    y_l = y_pred[:, 1]
    y_o = y_pred[:, 2]

    MSE = torch.mean((y_o - y_true) ** 2)  # Calculate MSE
    MPIW_p = torch.mean(torch.abs(y_u - y_o.detach()) + torch.abs(y_o.detach() - y_l))  # Calculate MPIW_penalty
    Constraints = (torch.exp(torch.mean(-y_u + y_o.detach()) + torch.max(torch.abs(y_o.detach() - y_true)).detach()) +
                   torch.exp(torch.mean(-y_o.detach() + y_l) + torch.max(torch.abs(y_o.detach() - y_true)).detach()) +
                   torch.exp(torch.mean(-y_u + y_l)))
    # Calculate loss
    Loss_S = MPIW_p + MSE * beta_[0] + Constraints * beta_[1]

    return Loss_S


def QD_plus_objective(y_pred, y_true, soften_=10, alpha_=0.05, beta_=None):
    """QD+ loss function, adapted from https://github.com/tarik/pi-snm-qde/blob/master/neural_pi/estimator/functions.py
    @param y_pred: NN output (y_u, y_l, y)
    @param y_true: Ground-truth.
    @param soften_: Softening factor used to approximate sigmoid function.
    @param alpha_: Prediction intervals capture samples with (1-alpha)% confidence.
    @param beta_: Array of hyperparameters: [lambda_1, lambda_2]."""
    # Separate upper and lower limits
    y_u = y_pred[:, 0]
    y_l = y_pred[:, 1]
    y_o = y_pred[:, 2]

    # Separate hyperparameters
    if beta_ is None:
        beta_ = [1, 1]
    lambda_1, lambda_2 = beta_
    ksi = 10  # According to the QD+ paper

    MSE = torch.mean((y_o - y_true) ** 2)  # Calculate MSE

    # Calculate soft captured vector, MPIW, and PICP
    K_SU = torch.sigmoid(soften_ * (y_u - y_true))
    K_SL = torch.sigmoid(soften_ * (y_true - y_l))
    K_S = torch.mul(K_SU, K_SL)
    MPIW_c = torch.sum(torch.mul((y_u - y_l), K_S)) / (torch.sum(K_S) + 0.0001)
    PICP_S = torch.mean(K_S)

    L_PICP = torch.pow(torch.relu((1. - alpha_) - PICP_S), 2) * 100  # PICP loss function

    # Calculate penalty function (Eq. 14 QD+ paper)
    Lp = torch.mean((torch.relu(y_l - y_o)) + (torch.relu(y_o - y_u)))

    # Calculate loss (Eq. 12 QD+ paper)
    Loss_S = (1 - lambda_1) * (1 - lambda_2) * MPIW_c + \
             lambda_1 * (1 - lambda_2) * L_PICP + \
             lambda_2 * MSE + ksi * Lp
    return Loss_S


def QD_objective(y_pred, y_true, soften_=1, alpha_=0.05, beta_=0.03, device='cuda:0'):
    """Original Loss_QD-soft, adapted from https://github.com/TeaPearce/Deep_Learning_Prediction_Intervals
    @param y_pred: CNN's output (y_u, y_l)
    @param y_true: Ground-truth.
    @param soften_: Softening factor used to approximate sigmoid function.
    @param alpha_: Prediction intervals capture samples with (1-alpha)% confidence.
    @param beta_: Importance factor width vs. coverage.
    @param device: Specify the device that is being used (cpu or gpu)."""
    # Separate upper and lower limits
    y_u = y_pred[:, 0]
    y_l = y_pred[:, 1]

    # Calculate hard captured vector
    K_HU = torch.max(torch.zeros(y_true.size()).to(device), torch.sign(y_u - y_true))
    K_HL = torch.max(torch.zeros(y_true.size()).to(device), torch.sign(y_true - y_l))
    K_H = torch.mul(K_HU, K_HL)

    # Calculate soft captured vector
    K_SU = torch.sigmoid(soften_ * (y_u - y_true))
    K_SL = torch.sigmoid(soften_ * (y_true - y_l))
    K_S = torch.mul(K_SU, K_SL)

    MPIW_c = torch.sum(torch.mul((y_u - y_l), K_H)) / (torch.sum(K_H) + 0.0001)
    PICP_S = torch.mean(K_S)

    # Calculate loss (Eq. 15 QD paper)
    Loss_S = MPIW_c + beta_ * len(y_true) / (alpha_ * (1 - alpha_)) * torch.max(torch.zeros(1).to(device),
                                                                                (1 - alpha_) - PICP_S)
    return Loss_S


#######################################################################################################################
# Class Definitions
#######################################################################################################################

class NNObject:
    """Helper class used to store the main information of a NN model."""

    def __init__(self, model, criterion, optimizer):
        self.network = model
        self.criterion = criterion
        self.optimizer = optimizer


class NNModel:

    def __init__(self, device, nfeatures, method):
        self.method = method
        self.device = device
        self.nfeatures = nfeatures
        self.basemodel = None  # AR_ARCC uses a base model trained only for target prediction

        if self.method in ['AQD', 'QD+']:
            self.output_size = 3
        elif self.method == 'AR_ARCC':
            self.output_size = 2
        elif self.method == 'QD':
            self.output_size = 2
        else:  # MC-Dropout
            self.output_size = 1

        criterion = nn.MSELoss()

        network = NN(input_shape=self.nfeatures, output_size=self.output_size)


        # network = NN(input_shape=self.nfeatures, output_size=self.output_size)

        network.to(self.device)
        # Training parameters
        optimizer = optim.Adadelta(network.parameters(),lr=0.1,rho=0.9)
        # optimizer = torch.optim.AdamW(network.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)

        self.model = NNObject(network, criterion, optimizer)

    def trainFold(self, Xtrain, Ytrain, Xval, Yval, batch_size, epochs, filepath, printProcess, yscale, alpha_=0.05):

        if self.method in ['AQD', 'MCDropout']:
            np.random.seed(7)
            random.seed(7)
            torch.manual_seed(7)
            torch.cuda.manual_seed(7)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        indexes = np.arange(len(Xtrain))
        np.random.shuffle(indexes)
        Xtrain = Xtrain[indexes]
        Ytrain = Ytrain[indexes]
        T = np.ceil(1.0 * len(Xtrain) / batch_size).astype(np.int32)  # 计算纪元中的步数


        val_mse = np.infty
        val_picp = 0
        val_mpiw = np.infty
        MPIWtr, PICPtr, MSEtr, MPIW, PICP, MSE, BETA = [], [], [], [], [], [], []
        widths = [0]
        picp, picptr, max_picptr, epoch_max_picptr = 0, 0, 0, 0
        first95 = True
        warmup = 10
        top = 1
        alpha_0 = alpha_
        err_prev, err_new, beta_, beta_prev, d_err = 0, 0, 1, 0, 1

        if self.method in ['AR_ARCC']:
            self.basemodel = NNModel(self.device, self.nfeatures, 'MCDropout')
            filepathbase = filepath.replace('AR_ARCC', 'MCDropout')
            if 'TuningResults' in filepathbase:
                filepathbase = filepathbase.replace('TuningResults', 'CVResults')
            try:
                self.basemodel.loadModel(filepathbase)
            except FileNotFoundError:

                sys.exit("The 'MCDropout' method needs be run first. It will generate the target-estimation NN 'f'")
            for target_param, param in zip(self.model.network.named_parameters(),
                                           self.basemodel.model.network.named_parameters()):
                if 'out' not in target_param[0]:
                    target_param[1].data.copy_(param[1].data)


        for epoch in trange(epochs):  # Epoch loop
            # Batch sorting
            if epoch > warmup and (self.method in ['AR_ARCC', 'QD', 'QD+']):
                indexes = np.argsort(widths)
            else:
                np.random.shuffle(indexes)

            # Shuffle batches
            list_inds = []
            for step in range(T):  # Batch loop
                # Generate indexes of the batch
                list_inds.append(indexes[step * batch_size:(step + 1) * batch_size])
            random.shuffle(list_inds)

            self.model.network.train()
            running_loss = 0.0
            for step, inds in enumerate(list_inds[:-1]):  # Batch loop
                # Get actual batches
                Xtrainb = torch.from_numpy(Xtrain[inds]).float().to(self.device)
                Ytrainb = torch.from_numpy(Ytrain[inds]).float().to(self.device)


                self.model.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model.network(Xtrainb)
                if self.method == 'AR_ARCC':
                    point_estimates = self.basemodel.model.network(Xtrainb)
                    if epoch > warmup:
                        loss = AR_ARCC_objective(outputs, Ytrainb, beta_=beta_, pe=point_estimates)
                    else:  # During the warmup period the objective is that \hat{y}^u = \hat{y}^l = \hat{y}
                        loss = AR_ARCC_objective(outputs, Ytrainb, beta_=None, pe=point_estimates)
                elif self.method == 'QD':
                    loss = QD_objective(outputs, Ytrainb, device=self.device, beta_=alpha_)
                elif self.method == 'QD+':
                    loss = QD_plus_objective(outputs, Ytrainb, beta_=alpha_)
                else:
                    outputs = outputs.squeeze(1)
                    loss = self.model.criterion(outputs, Ytrainb)
                loss.backward()
                self.model.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if printProcess and epoch % 10 == 0:
                    print('[%d, %5d] loss: %.5f' % (epoch + 1, step + 1, loss.item()))

            # Validation step
            with torch.no_grad():
                self.model.network.eval()
                if self.method in ['AR_ARCC', 'MCDropout']:  # These methods use mult forward passes w active dropout
                    samples = 50
                    ypredtr = self.evaluateFoldUncertainty(valxn=Xtrain, batch_size=len(Xtrain), MC_samples=samples)
                    ypred = self.evaluateFoldUncertainty(valxn=Xval, batch_size=len(Xval), MC_samples=samples)
                    ypredtr = np.mean(ypredtr, axis=2)
                    ypred = np.mean(ypred, axis=2)
                else:
                    ypredtr = self.model.network(torch.from_numpy(Xtrain).float().to(self.device)).cpu().numpy()
                    ypred = self.model.network(torch.from_numpy(Xval).float().to(self.device)).cpu().numpy()
                # Reverse normalization
                Ytrain_original = utils.reverseMinMaxScale(Ytrain, yscale[0], yscale[1])
                Yval_original = utils.reverseMinMaxScale(Yval, yscale[0], yscale[1])
                ypredtr = utils.reverseMinMaxScale(ypredtr, yscale[0], yscale[1])
                ypred = utils.reverseMinMaxScale(ypred, yscale[0], yscale[1])

                ##################################################
                # Calculate metrics
                ##################################################
                # Calculate MSE
                if self.method in ['AR_ARCC', 'QD+']:
                    msetr = utils.mse(Ytrain_original, ypredtr[:, 2])
                    mse = utils.mse(Yval_original, ypred[:, 2])
                elif self.method == 'QD':
                    msetr = utils.mse(Ytrain_original, (ypredtr[:, 0] + ypredtr[:, 1]) / 2)
                    mse = utils.mse(Yval_original, (ypred[:, 0] + ypred[:, 1]) / 2)
                else:
                    msetr = utils.mse(Ytrain_original, ypredtr[:, 0])
                    mse = utils.mse(Yval_original, ypred[:, 0])
                MSEtr.append(msetr)
                MSE.append(mse)
                if self.method in ['AR_ARCC', 'QD', 'QD+']:
                    # Calculate MPIW and PICP
                    y_true = torch.from_numpy(Ytrain_original).float().to(self.device)
                    y_utr = torch.from_numpy(ypredtr[:, 0]).float().to(self.device)
                    y_ltr = torch.from_numpy(ypredtr[:, 1]).float().to(self.device)
                    K_U = torch.max(torch.zeros(y_true.size()).to(self.device), torch.sign(y_utr - y_true))
                    K_L = torch.max(torch.zeros(y_true.size()).to(self.device), torch.sign(y_true - y_ltr))
                    Ktr = torch.mul(K_U, K_L)
                    picptr = torch.mean(Ktr).item()
                    y_true = torch.from_numpy(Yval_original).float().to(self.device)
                    y_u = torch.from_numpy(ypred[:, 0]).float().to(self.device)
                    y_l = torch.from_numpy(ypred[:, 1]).float().to(self.device)
                    K_U = torch.max(torch.zeros(y_true.size()).to(self.device), torch.sign(y_u - y_true))
                    K_L = torch.max(torch.zeros(y_true.size()).to(self.device), torch.sign(y_true - y_l))
                    K = torch.mul(K_U, K_L)
                    # Update curves
                    MPIWtr.append((torch.sum(torch.mul((y_utr - y_ltr), Ktr)) / (torch.sum(Ktr) + 0.0001)).item())
                    PICPtr.append(picptr)
                    width = (torch.sum(torch.mul((y_u - y_l), K)) / (torch.sum(K) + 0.0001)).item()
                    picp = torch.mean(K).item()
                    MPIW.append(width)
                    PICP.append(picp)

                    widths = (y_utr - y_ltr).cpu().numpy()
            ##################################################
            # Save model if there's improvement
            ##################################################

            if self.method in ['AR_ARCC', 'QD+', 'QD']:

                if (((val_picp == picp < .95 and width < val_mpiw) or (val_picp < picp < .95)) and first95) or \
                        (picp >= 0.9499 and first95) or \
                        (picp >= 0.9499 and width < val_mpiw and not first95):  # and val_std < std
                    if picp >= .9499:
                        first95 = False
                    val_mse = mse
                    val_picp = picp
                    val_mpiw = width
                    if filepath is not None:
                        torch.save(self.model.network.state_dict(), filepath)
            else:  # Save model if MSE decreases
                if mse < val_mse:
                    val_mse = mse
                    if filepath is not None:
                        torch.save(self.model.network.state_dict(), filepath)


            # Check if picp has reached convergence
            if epoch >= warmup and self.method == 'AR_ARCC':
                if picptr > max_picptr:
                    max_picptr = picptr
                    epoch_max_picptr = epoch
                else:
                    if epoch == epoch_max_picptr + 500 and \
                            picptr <= max_picptr:  # If 500 epochs have passed without increasing PICP
                        top = .95
                        # first100 = True
                        # alpha_0 = alpha_ / 2
                # if picptr >= 0.999 and not first100:
                #     first100 = True
                #     top = .95
                # alpha_0 = alpha_ / 2
                # Beta hyperparameter
                err_new = top - picptr
                beta_ = beta_ + alpha_0 * err_new
                # Update parameters
                BETA.append(beta_)

            ##################################################
            # Print
            ##################################################
            # Print every 10 epochs
            if printProcess and epoch % 10 == 0:
                if self.method == 'MCDropout':
                    print('VALIDATION: Training_MSE: %.5f. Best_MSE: %.5f' % (msetr, val_mse))
                else:
                    print('VALIDATION: Training_MSE: %.5f. Best_MSEval: %.5f. MSE val: %.5f. PICP val: %.5f. '
                          'MPIW val: %.5f' % (msetr, val_mse, mse, picp, width))
                    print(val_picp)
                    print(val_mpiw)
                    print(picptr)
                    print(beta_)
                    print(top)

        # Save training metrics
        if filepath is not None:
            with open(filepath + '_validationMSE', 'wb') as fil:
                pickle.dump(val_mse, fil)
            # Save history
            np.save(filepath + '_historyMSEtr', MSEtr)
            np.save(filepath + '_historyMSE', MSE)
            if 'QD' in self.method:  # Average upper and lower limit to obtain expected output
                np.save(filepath + '_historyMPIWtr', MPIWtr)
                np.save(filepath + '_historyPICPtr', PICPtr)
                np.save(filepath + '_historyMPIW', MPIW)
                np.save(filepath + '_historyPICP', PICP)


        return MPIW, PICP, MSE, val_mse, val_picp, val_mpiw

    def evaluateFold(self, valxn, maxs, mins, batch_size):
        """Retrieve point predictions."""
        if maxs is not None and mins is not None:
            valxn = utils.reverseMinMaxScale(valxn, maxs, mins)

        ypred = []
        with torch.no_grad():
            self.model.network.eval()
            Teva = np.ceil(1.0 * len(valxn) / batch_size).astype(np.int32)
            indtest = np.arange(len(valxn))
            for b in range(Teva):
                inds = indtest[b * batch_size:(b + 1) * batch_size]
                if self.method == 'AQD':
                    ypred_batch = self.model.network(torch.from_numpy(valxn[inds]).float().to(self.device))
                    ypred_batch = ypred_batch[:, 2]
                elif self.method == 'AR_ARCC':
                    ypred_batch = self.basemodel.model.network(torch.from_numpy(valxn[inds]).float().to(self.device))
                elif self.method == 'QD':
                    ypred_batch = self.model.network(torch.from_numpy(valxn[inds]).float().to(self.device))
                    ypred_batch = (ypred_batch[:, 0] + ypred_batch[:, 1]) / 2
                ypred = ypred + (ypred_batch.cpu().numpy()).tolist()

        return ypred

    def evaluateFoldUncertainty(self, valxn, maxs=None, mins=None, batch_size=512, MC_samples=50):
        """Retrieve point predictions and PIs"""
        np.random.seed(7)  # Initialize seed to get reproducible results
        random.seed(7)
        torch.manual_seed(7)
        torch.cuda.manual_seed(7)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if maxs is not None and mins is not None:
            valxn = utils.reverseMinMaxScale(valxn, maxs, mins)

        with torch.no_grad():
            preds_MC = np.zeros((len(valxn), MC_samples))
            if self.method in ["AR_ARCC", "QD", "QD+"]:
                preds_MC = np.zeros((len(valxn), 3, MC_samples))
            for it in range(0, MC_samples):  # Test the model 'MC_samples' times
                ypred = []
                self.model.network.eval()
                if self.method in ["AR_ARCC", "MCDropout"]:  # Only these methods activate Dropout layers during test
                    enable_dropout(self.model.network)  # Set Dropout layers to test mode
                Teva = np.ceil(1.0 * len(valxn) / batch_size).astype(np.int32)  # Number of batches
                indtest = np.arange(len(valxn))
                for b in range(Teva):
                    inds = indtest[b * batch_size:(b + 1) * batch_size]
                    ypred_batch = self.model.network(torch.from_numpy(valxn[inds]).float().to(self.device))
                    if self.method == "AR_ARCC":
                        self.basemodel.model.network.eval()
                        enable_dropout(self.basemodel.model.network)
                        ypred_batch = ypred_batch.cpu().numpy()
                        ypred_batchtmp = np.zeros((ypred_batch.shape[0], 3))
                        pe_batch = self.basemodel.model.network(torch.from_numpy(valxn[inds]).float().to(self.device))
                        pe_batch = pe_batch.cpu().numpy()
                        ypred_batchtmp[:, :2] = ypred_batch
                        ypred_batchtmp[:, 2] = pe_batch.squeeze(1)
                        ypred_batch = ypred_batchtmp
                    elif self.method == "QD":
                        ypred_batch = ypred_batch.cpu().numpy()
                        ypred_batchtmp = np.zeros((ypred_batch.shape[0], 3))
                        ypred_batchtmp[:, 0] = ypred_batch[:, 0]
                        ypred_batchtmp[:, 1] = ypred_batch[:, 1]
                        ypred_batch = ypred_batchtmp
                        ypred_batch[:, 2] = (ypred_batch[:, 0] + ypred_batch[:, 1]) / 2
                    elif self.output_size == 1:
                        ypred_batch = ypred_batch.squeeze(1)
                    ypred = ypred + ypred_batch.tolist()

                if self.method in ["AR_ARCC", "QD", "QD+"]:
                    preds_MC[:, :, it] = np.array(ypred)
                else:
                    preds_MC[:, it] = np.array(ypred)
        return preds_MC

    def loadModel(self, path):
        # self.model.network.load_state_dict(torch.load(path),False)

        def load_matching_weights(model, pretrained_weights):
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_weights.items() if
                               k in model_dict and model_dict[k].size() == v.size()}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

        pretrained_weights = torch.load(path)
        load_matching_weights(self.model.network, pretrained_weights)


        if self.method in ['AR_ARCC']:
            self.basemodel = NNModel(self.device, self.nfeatures, 'MCDropout')
            filepathbase = path.replace('AR_ARCC', 'MCDropout')
            if 'TuningResults' in filepathbase:
                filepathbase = filepathbase.replace('TuningResults', 'CVResults')
            self.basemodel.loadModel(filepathbase)
