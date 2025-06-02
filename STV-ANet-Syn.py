import argparse
import numpy as np
import os
import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.utils.data import DataLoader, Dataset
import cvxpy as cp
from UnmixingUtils import UnmixingUtils


class L1NMF_Net(nn.Module):
    def __init__(self, layerNum, M, A):
        super(L1NMF_Net, self).__init__()
        R = np.size(M, 1)
        eig, _ = np.linalg.eig(M.T @ M)
        eig += 0.1
        L = 1 / np.max(eig)
        lambd = np.ones((1, R)) * 0.01 * L
        eig, _ = np.linalg.eig(A @ A.T)
        eig += 0.1
        L2 = np.max(eig)
        L2 = 1 / L2

        self.p = nn.ParameterList()
        self.L = nn.ParameterList()
        self.lambd = nn.ParameterList()
        self.L2 = nn.ParameterList()
        self.W_a = nn.ParameterList()
        self.alpha = nn.ParameterList()
        self.tau = nn.ParameterList()
        self.layerNum = layerNum
        temp = self.calW(M)
        for k in range(self.layerNum):
            self.L.append(nn.Parameter(torch.FloatTensor([L])))
            self.L2.append(nn.Parameter(torch.FloatTensor([L2])))
            self.lambd.append(nn.Parameter(torch.FloatTensor(lambd)))
            self.p.append(nn.Parameter(torch.FloatTensor([0.5])))
            self.alpha.append(nn.Parameter(torch.FloatTensor([10000])))
            self.tau.append(nn.Parameter(torch.FloatTensor([0.5])))
            self.W_a.append(nn.Parameter(torch.FloatTensor(temp)))
        self.layerNum = layerNum

    def forward(self, X, _M, _A, im_size):
        M = list()
        M.append(torch.FloatTensor(_M))
        A = list()
        A.append(torch.FloatTensor(_A.T))
        B = A[-1]
        for k in range(self.layerNum):
            # Update A
            _A = self.update_A(X, M[-1], A[-1], B, self.W_a[k], self.lambd[k], self.alpha[k], self.p[k],
                               self.L[k])
            A.append(_A)

            # Update B
            B = self.update_B(A[-1], self.alpha[k], self.tau[k], im_size)

            # Update M
            _M = self.update_M(X, M[-1], A[-1], self.L2[k])
            M.append(_M)

        return M, A

    def update_B(self, A, alpha, tau, im_size):
        B = A
        for i in range(A.shape[0]):
            temp = A[i].reshape(*im_size)
            z = self.prox_TV(temp, 2 * tau / alpha)
            B[i] = z.reshape(im_size[0] * im_size[1])
        B = F.relu(B)
        B = self.sum2one(B)
        return B

    def update_A(self, X, M, A, B, W, lambd, alpha, p, L):
        lam = lambd.repeat(A.size(1), 1).T
        grad = W.T.mm(M @ A - X) + alpha * (A - B)
        A = A - L * grad
        A = self.self_active(A, p, lam)
        A = F.relu(A)
        A = self.sum2one(A)
        return A

    def update_M(self, X, M, A, L):
        grad = (M @ A - X) @ A.T
        M = M - L * grad
        M = F.relu(M)
        return M

    def prox_TV(self, b, lmbda, **kwargs):
        # Optional input arguments
        if 'rel_obj' not in kwargs:
            kwargs['rel_obj'] = 1e-4
        if 'verbose' not in kwargs:
            kwargs['verbose'] = 1
        if 'max_iter' not in kwargs:
            kwargs['max_iter'] = 2
        # Initializations
        r, s = self.gradient_op(torch.zeros_like(b))
        pold, qold = r.clone(), s.clone()
        told = 1
        prev_obj = 0
        # Main iterations
        # if kwargs['verbose'] > 1:
        #     print('  Proximal TV operator:\n')
        for iter in range(1, kwargs['max_iter'] + 1):
            # Current solution
            sol = b - lmbda * self.div_op(r, s)
            # Objective function value
            obj = 0.5 * torch.linalg.norm(b.T.ravel() - sol.T.ravel()) ** 2 + lmbda * self.TV_norm(sol)
            rel_obj = torch.abs(obj - prev_obj) / obj
            prev_obj = obj
            # Stopping criterion
            # if kwargs['verbose'] > 1:
            #     print(f'   Iter {iter}, obj = {obj}, rel_obj = {rel_obj}')
            if rel_obj < kwargs['rel_obj']:
                crit_TV = 'TOL_EPS'
                break
            # Update divergence vectors and project
            dx, dy = self.gradient_op(sol)
            r = r - 1 / (8 * lmbda) * dx
            s = s - 1 / (8 * lmbda) * dy
            weights = torch.maximum(torch.ones_like(s), torch.sqrt(torch.abs(r) ** 2 + torch.abs(s) ** 2 + 1e-8))
            p = r / weights
            q = s / weights
            # FISTA update
            t = (1 + np.sqrt(4 * told ** 2)) / 2
            r = p + (told - 1) / t * (p - pold)
            pold = p.clone()
            s = q + (told - 1) / t * (q - qold)
            qold = q.clone()
            told = t.copy()
        # Log after the minimization
        if 'crit_TV' not in locals():
            crit_TV = 'MAX_IT'
        # if kwargs['verbose'] >= 1:
        #     print(f'  Prox_TV: obj = {obj}, rel_obj = {rel_obj}, {crit_TV}, iter = {iter}')
        return sol

    def gradient_op(self, I, **weights):
        dx = torch.cat((I[1:, :] - I[:-1, :], torch.zeros((1, I.shape[1]))), dim=0)
        dy = torch.cat((I[:, 1:] - I[:, :-1], torch.zeros((I.shape[0], 1))), dim=1)
        if 'dx' in weights:
            dx = dx * weights['dx']
        if 'dy' in weights:
            dy = dx * weights['dy']
        return dx, dy

    def div_op(self, dx, dy, **weights):
        if 'dx' in weights:
            dx = dx * torch.conj(weights['dx'])
        if 'dy' in weights:
            dx = dx * torch.conj(weights['dy'])
        I = torch.cat((dx[[0], :], dx[1:-1, :] - dx[:-2, :], -dx[[-2], :]), dim=0)
        I = I + torch.cat((dy[:, [0]], dy[:, 1:-1] - dy[:, :-2], -dy[:, [-2]]), dim=1)
        return I

    def TV_norm(self, u):
        dx, dy = self.gradient_op(u)
        temp = torch.sqrt(torch.abs(dx) ** 2 + torch.abs(dy) ** 2 + 1e-8)
        y = torch.sum(temp)
        return y

    def calW(self, D):
        (m, n) = D.shape
        W = cp.Variable(shape=(m, n))
        obj = cp.Minimize(cp.norm(W.T @ D, 'fro'))
        # Create two constraints.
        constraint = [cp.diag(W.T @ D) == 1]
        prob = cp.Problem(obj, constraint)
        result = prob.solve(solver=cp.SCS, max_iters=1000)
        print('residual norm {}'.format(prob.value))
        # print(W.value)
        return W.value

    def half_thresholding(self, z_hat, mu):
        c = pow(54, 1 / 3) / 4
        tau = z_hat.abs() - c * pow(mu, 2 / 3)
        v = z_hat
        ind = tau > 0
        v[ind] = 2 / 3 * z_hat[ind] * (
                1 + torch.cos(2 * math.pi / 3 - 2 / 3 * torch.acos(mu[ind] / 8 * pow(z_hat[ind].abs() / 3, -1.5))))
        v[tau < 0] = 0
        return v

    def soft_thresholding(self, z_hat, mu):
        return z_hat.sign() * F.relu(z_hat.abs() - mu)

    def self_active(self, x, p, lam):
        tau = pow(2 * (1 - p) * lam, 1 / (2 - p)) + p * lam * pow(2 * lam * (1 - p), (p - 1) / (2 - p))
        v = x
        ind = (x - tau) > 0
        ind2 = (x - tau) <= 0
        v[ind] = x[ind].sign() * (x[ind].abs() - p * lam[ind] * pow(x[ind].abs(), p - 1))
        v[ind2] = 0
        v[v > 1] = 1
        return v

    def sum2one(self, Z):
        temp = Z.sum(0)
        temp = temp.repeat(Z.size(0), 1) + 0.0001
        return Z / temp


class RandomDataset(Dataset):
    def __init__(self, data, label, length):
        self.data = data
        self.len = length
        self.label = label

    def __getitem__(self, item):
        return torch.Tensor(self.data[:, item]).float(), torch.Tensor(self.label[:, item]).float()

    def __len__(self):
        return self.len


def prepare_data(dataFile):
    data = scio.loadmat(dataFile)
    X = data['Y_noise']
    A = data['A_true']
    s = data['S_true'].T
    A0 = data['A_est']
    S0 = data['S_est']
    image_size = (64,64)
    return X, A, s, A0, S0, image_size


def prepare_train(X, s, trainFile):
    train_index = scio.loadmat(trainFile)
    train_index = train_index['train']
    train_index = train_index - 1
    train_data = np.squeeze(X[:, train_index])
    train_labels = np.squeeze(s[:, train_index])
    nrtrain = np.size(train_index, 1)
    train_size = (int(nrtrain / 10), 10)
    return train_data, train_labels, nrtrain, train_size


def prepare_init(initFile):
    init = scio.loadmat(initFile)
    A0 = init['Cn']
    S0 = init['o'][0, 0]['S']
    return A0, S0


def set_param(layerNum, lr, lrD, batch_size=4096):
    parser = argparse.ArgumentParser(description="LISTA-Net")
    parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
    parser.add_argument('--end_epoch', type=int, default=500, help='epoch number of end training')
    parser.add_argument('--layer_num', type=int, default=layerNum, help='phase number of ISTA-Net')
    parser.add_argument('--learning_rate_decoder', type=float, default=lrD, help='learning rate for decoder')
    parser.add_argument('--learning_rate', type=float, default=lr, help='learning rate')
    parser.add_argument('--batch_size', type=float, default=batch_size, help='batch size')
    parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
    parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
    parser.add_argument('--log_dir', type=str, default='log', help='log directory')
    args = parser.parse_args()
    return args


def train(lrD, layerNum, lr, train_data, test_data, nrtrain, A0, S0, X, A, s, SNR, train_size, image_size):
    batch_size = nrtrain
    args = set_param(layerNum, lr, lrD, batch_size=batch_size)
    model_dir = "./%s/SNR_%sSNMF_layer_%d_lr_%.8f_lrD_%.8f" % (
        args.model_dir, SNR, args.layer_num, args.learning_rate, args.learning_rate_decoder)
    log_file_name = "./%s/SNR_%sSNMF_layer_%d_lr_%.8f_lrD_%.8f.txt" % (
        args.log_dir, SNR, args.layer_num, args.learning_rate, args.learning_rate_decoder)
    model = L1NMF_Net(args.layer_num, A0, S0)
    criterion = nn.MSELoss(reduction='sum')
    trainloader = DataLoader(dataset=RandomDataset(train_data, test_data, nrtrain), batch_size=args.batch_size,
                             num_workers=0,
                             shuffle=False)
    learning_rate = args.learning_rate
    learning_rate_decoder = args.learning_rate_decoder
    opt = optim.Adam([{'params': [L_a for L_a in model.L] + [p for p in model.p] + [a for a in model.alpha] + [t for t
                                                                                                               in
                                                                                                               model.tau]},
                      {'params': [L_b for L_b in model.L2] + [W_a_ for W_a_ in model.W_a] + [the for the in
                                                                                             model.lambd],
                       'lr': learning_rate_decoder}],
                     lr=learning_rate, weight_decay=0.001, betas=(0.9, 0.9))
    start_epoch = args.start_epoch
    end_epoch = args.end_epoch
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    running_loss = 0.0
    last_loss = 1
    for epoch_i in range(start_epoch + 1, end_epoch + 1):
        if epoch_i <= 5 and epoch_i % 2 == 0:
            learning_rate = learning_rate / 25
            opt = optim.Adam([{'params': [L_a for L_a in model.L] + [p for p in model.p] + [a for a in model.alpha] + [t
                                                                                                                       for
                                                                                                                       t
                                                                                                                       in
                                                                                                                       model.tau]},
                              {'params': [L_b for L_b in model.L2] + [W_a_ for W_a_ in model.W_a] + [the for the in
                                                                                                     model.lambd],
                               'lr': learning_rate_decoder}],
                             lr=learning_rate, weight_decay=0.001, betas=(0.9, 0.9))
        if epoch_i > 100 and epoch_i % 50 == 0:
            learning_rate = learning_rate / 1.5
            learning_rate_decoder = learning_rate_decoder / 1.5
            opt = optim.Adam([{'params': [L_a for L_a in model.L] + [p for p in model.p] + [a for a in model.alpha] + [t
                                                                                                                       for
                                                                                                                       t
                                                                                                                       in
                                                                                                                       model.tau]},
                              {'params': [L_b for L_b in model.L2] + [W_a_ for W_a_ in model.W_a] + [the for the in
                                                                                                     model.lambd],
                               'lr': learning_rate_decoder}],
                             lr=learning_rate, weight_decay=0.001, betas=(0.9, 0.9))
        for data_batch in trainloader:
            batch_x, batch_label = data_batch
            output_end, output_abun = model(batch_x.T, A0, batch_label, train_size)
            loss = sum(
                [criterion(output_end[i + 1] @ output_abun[i + 1], batch_x.T) for i in range(layerNum)]) / layerNum
            opt.zero_grad()
            loss.backward()
            opt.step()
        for i in range(layerNum):
            t1 = model.p[i].data
            t1[t1 < 0] = 1e-4
            t1[t1 > 1] = 1
            model.p[i].data.copy_(t1)
            t3 = model.tau[i].data
            t3[t3 < 0] = 1e-5
            t3[t3 > 1] = 1
            model.tau[i].data.copy_(t3)
            running_loss += loss.item()
        temp = abs(running_loss - last_loss) / last_loss
        output_data = 'train===epoch: %d, loss:  %.5f, tol: %.6f\n' % (epoch_i, running_loss, temp)
        print(output_data)
        last_loss = running_loss
        running_loss = 0.0
        if epoch_i % 5 == 0:
            torch.save(model, "%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters
    util = UnmixingUtils(A, s.T)
    out1, out2 = model(torch.FloatTensor(X), A0, S0.T, image_size)
    Distance, meanDistance, sor = util.hyperSAD(out1[-1].detach().numpy())
    rmse = util.hyperRMSE(out2[-1].T.detach().numpy(), sor)
    output_data = 'Res: SAD: %.5f RMSE:  %.5f' % (meanDistance, rmse)
    print(output_data)

    # Create a folder for saving data
    dir_name = "Result"
    save_dir = os.path.join(os.getcwd(), dir_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_name = os.path.join(save_dir,
                             f"Syn25dB_STV-ANet_lN{layerNum}_lr{lr}_lrD{lrD}")
    # save MAT
    scio.savemat(file_name + '.mat', {'A_est': out1[-1].to('cpu').detach().numpy(),
                                      'S_est': out2[-1].to('cpu').detach().numpy(),
                                      'A_true': A,
                                      'S_true': s,
                                      'SAD': meanDistance,
                                      'RMSE': rmse})

    return meanDistance, rmse


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataFile = 'Syn25dB.mat'
    trainFile = 'train_4096_500.mat'
    X, A, s, A0, S0, image_size = prepare_data(dataFile)
    train_data, train_labels, nrtrain, train_size = prepare_train(X, S0, trainFile)
    layerNum = 30
    lr = 0.69
    lrD = 1e-6
    train(lrD=lrD, lr=lr, layerNum=layerNum, train_data=train_data, test_data=train_labels, nrtrain=nrtrain,
          A0=A0, S0=S0, X=X, A=A, s=s.T, SNR='25dB', train_size=train_size, image_size=image_size)

