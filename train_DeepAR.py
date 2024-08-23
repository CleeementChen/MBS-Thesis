"""Sample program to train and validate with DeepAR iteratively"""
import sys
import torch
import torch.nn as nn
import argparse
import torch.nn.functional as F
import torch.utils.data as data_utils
import numpy as np
from os import path
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from Helper import *
from DeepAR import DeepAR
from DeepAR import loss_deepAR


# Index of the output feature to forecast
SALIENT_IND = 10
# Forecast length, if different from decoder length
N_TESTING = 5
Data = './Training_Data.npy'        # Path to training data
Target = './Testing_Data.npy'       # Path to validation data

def main(args):
    """Primary trainig and testing loop. Validates by using the expected value of
    the distribution.

    Inputs:
    
        - Model parameters via CLI
        - Path to numpy datasets, and related values (see above)
        
    Outputs:
    
        - Model files for the best and latest epochs (as `pth`)
        - CSV of best epoch and MSE

    """
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pin_memory = True if torch.cuda.is_available() else False

    # Prepare datasets
    #
    # Data is of the form (sample_num, time_step, feat_val) as a numpy array
    # Target is of the form (sample_num, target_val) as a numpy array
    #
    # For each sample, given the whole series, we predict the last time-step
    Data = np.load(Data)
    Target = np.load(Target)

    # Split the data 90/10 for train/test
    # Batch size must divide 900 and 100
    TrainingData = torch.as_tensor(Data[:900,:,:], device = device, dtype = torch.float)
    TrainingTarget = torch.as_tensor(Target[:900,:], device = device, dtype = torch.float)
    TestingData = torch.as_tensor(Data[900:,:,:], device = device, dtype = torch.float)
    TestingTarget = torch.as_tensor(Target[900:,:], device = device, dtype = torch.float)

    train_set = torch.utils.data.TensorDataset(TrainingData, TrainingTarget)
    train_loader = torch.utils.data.DataLoader(train_set, 
        batch_size=args.batch_size, shuffle=True)
    test_set = torch.utils.data.TensorDataset(TestingData, TestingTarget)
    test_loader = torch.utils.data.DataLoader(test_set, 
        batch_size=args.batch_size, shuffle=False)

    Results = np.zeros((2, 2), dtype=object)
    Results[0, :] = ["Best Epoch", "Best MSE"]


    # Initialize DeepAR model with default LSTM
    net = DeepAR(args.input_size, device, encoder_len = args.enc_len, decoder_len = args.dec_len, 
        hidden_size = args.hidden_size, num_layers = args.num_layers, dropout = args.rnndropout)
    net.to(device = device)

    criterion = loss_deepAR
    optimizerTimeAtten = torch.optim.Adam(net.parameters(), 
        lr = args.learning_rate)


    # Some arbitrary model name for the most recent model, and the best performing
    # one (based on validation set error)
    modelName = "./SampleModel"
    saveModelBestName = modelName + "_BEST.pth"
    saveModelLastName = modelName + "_LAST.pth"

    # Load a previous state?
    if (args.load and path.isfile(saveModelLastName)):
        net.load_state_dict(torch.load(saveModelLastName))

    # Train for specified epochs or until convergence (best epoch is greater than 
    # 250 epochs back.
    test_best_MSE = np.inf
    best_epoch = 0
    for epoch in range(args.num_epochs):
        net.train()
        running_loss = 0
        running_loss_count = 0
        #
        # Training
        #
        # `samples` is of the form (batch_size, tot_time_steps, tot_features)
        for _, (samples, labels) in enumerate(train_loader):
            # Zero the encoder hidden and cell states
            hidden_encoder = torch.zeros(net.rnn_layers, args.batch_size, 
                net.rnn_hidden_size, device = device)
            cell_encoder = torch.zeros(net.rnn_layers, args.batch_size, 
                net.rnn_hidden_size, device = device)
            seq_len = samples.size()[1]

            # breaks prematurely if the decoder extends past available
            # data
            for idx in range(seq_len):
                # Record the number of seen values
                running_loss_count += args.batch_size * net.encoder_len

                # Shift encoder input, insert the most recent to decoder input
                encoder_input = samples[:, idx : (net.encoder_len + idx), :].clone().detach().requires_grad_(True).to(device = device)
                # When forwarding the input by 1, discard the first value that rollovers to the last spot
                decoder_input = torch.roll(samples, -1, 1)[:, :-1, :]
                decoder_input = decoder_input[:, idx : (net.decoder_len + idx), SALIENT_IND].clone().detach().requires_grad_(True).to(device = device)
                # Would be unnecessary for an output with more than one feature.
                # But here, the output is compressed by indexing
                decoder_input = torch.unsqueeze(decoder_input, 2)

                
                if (encoder_input.size()[1] < net.encoder_len or \
                    decoder_input.size()[1] < net.decoder_len):
                    break

                # Forward pass
                optimizerTimeAtten.zero_grad()
                outputs, mu_collection, sigma_collection, \
                    hidden_encoder, cell_encoder = net(encoder_input, 
                    decoder_input, hidden_encoder, cell_encoder)
                loss = criterion(mu_collection, sigma_collection, decoder_input)
                running_loss += loss.item()
                loss.backward()
                optimizerTimeAtten.step()


                hidden_encoder.detach_()
                cell_encoder.detach_()
        #
        # Testing
        #
        # The MSE is calculated between the __expected value__ of the distribution and target values
        #
        net.eval()
        test_count = 0
        test_total_MSE = 0
        # samples is of the form (batch_size, tot_time_steps, tot_features)
        for _, (samples, labels) in enumerate(test_loader):
            encoder_input = samples[:, -(net.encoder_len + 1) : -1, :].clone().detach().to(device = device)
            decoder_input = samples[:, -1:, SALIENT_IND].clone().detach().to(device = device)
            decoder_input = torch.unsqueeze(decoder_input, 2)
            # Returns decoder_len outputs
            outputs, mu_collection, sigma_collection, \
                _, _ = net(encoder_input, decoder_input)
            # No divison in mse_loss
            test_mse = mse_loss(mu_collection[:, :N_TESTING, 0], labels.to(device = device))
            test_total_MSE += test_mse
            test_count += N_TESTING * samples.size(0)
        test_MSE = test_total_MSE / (test_count * 1.0)

        # Update results or break?
        if (test_MSE < test_best_MSE):
            test_best_MSE = test_MSE
            best_epoch = epoch + 1
            torch.save(net.state_dict(), saveModelBestName)
        if ((epoch + 1) % 30 == 0):
            torch.save(net.state_dict(), saveModelLastName)
        if (epoch + 1 - best_epoch >= 250):
            torch.save(net.state_dict(), saveModelBestName)
            break

        print ('{}, DeepAR: Epoch {}, Av. Run Loss: {:.5f}, Best test MSE of {:.5f} at epoch {}' 
               .format(modelName, epoch+1, running_loss / (running_loss_count * 1.0), test_best_MSE, best_epoch))
    print("Saving...")
    Results[1, :] = [best_epoch, test_best_MSE]
    save_intoCSV(Results, './Results_MSE.csv')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--enc_len', type=int, default=8)
    parser.add_argument('--dec_len', type=int, default=N_TESTING)
    parser.add_argument('--input_size', type=int, default=25)
    parser.add_argument('--hidden_size', type=int, default=20)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=180)
    parser.add_argument('--num_epochs', type=int, default=5000)
    parser.add_argument('--learning_rate', type=float, default=0.25)
    parser.add_argument('--rnndropout', type=float, default=0.3)
    # Load most recent `pth` model?
    parser.add_argument('--load', type=bool, default=False)
    return  parser.parse_args()

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
