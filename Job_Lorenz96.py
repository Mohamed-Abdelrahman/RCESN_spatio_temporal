import numpy as np
import scipy.sparse as sparse
import pandas as pd
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
import matplotlib.pyplot as plt
from collections import OrderedDict

# global variables

#Default initial condition, it defines the position in the data where the ESN starts the training phase
shift_k = 0

# Reservoir size of the ESN
approx_res_size = 5000

# ESN parameters used for training and predictions
res_params = {'radius': 0.1,
              'degree': 3,
              'sigma': 0.5,
              'train_length': 100000,
              'D': approx_res_size,
              'num_inputs': 8,
              'predict_length': 2000,
              'beta': 0.0001
              }

# Data file should be in the same folder as this code
dataf = pd.read_csv('3tier_lorenz_v3.csv', header=None)
data = np.transpose(np.array(dataf))
measurements = data

# Path to be used for saving figure and files, and accessing necessary input files
#path = "Plots/96/"+str(approx_res_size)+"/"+"Prediction"+"/"
path = "Plots/96/"+str(approx_res_size)+"/"+"Clustering"+"/"+str(shift_k)+"/"

# The ESN functions for training
def generate_reservoir(size, radius, degree):
    sparsity = degree / float(size)
    A = sparse.rand(size, size, density=sparsity).todense()
    vals = np.linalg.eigvals(A)
    e = np.max(np.abs(vals))  #
    A = (A / e) * radius  # To make sure the max eigen value <= 1
    return A

def reservoir_layer(A, Win, input, res_params):
    states = np.zeros((res_params['D'], res_params['train_length']))
    for i in range(res_params['train_length'] - 1):
        states[:, i + 1] = np.tanh(np.dot(A, states[:, i]) + np.dot(Win, input[:, i]))
    return states

def train_reservoir(res_params, data):
    A = generate_reservoir(res_params['D'], res_params['radius'], res_params['degree'])
    q = int(res_params['D'] / res_params['num_inputs'])
    Win = np.zeros((res_params['D'], res_params['num_inputs']))
    for i in range(res_params['num_inputs']):
        np.random.seed(seed=i)
        Win[i * q: (i + 1) * q, i] = res_params['sigma'] * (-1 + 2 * np.random.rand(1, q)[0])

    states = reservoir_layer(A, Win, data, res_params)
    Wout = train(res_params, states, data)
    x = states[:, -1]
    return x, Wout, A, Win, states

def train_reservoir_different_seed(res_params, data, seed_index):
    A = generate_reservoir(res_params['D'], res_params['radius'], res_params['degree'])
    q = int(res_params['D'] / res_params['num_inputs'])
    Win = np.zeros((res_params['D'], res_params['num_inputs']))
    for i in range(res_params['num_inputs']):
        np.random.seed(seed=(i+1)*seed_index+seed_index%(i+1))

        Win[i * q: (i + 1) * q, i] = res_params['sigma'] * (-1 + 2 * np.random.rand(1, q)[0])

    states = reservoir_layer(A, Win, data, res_params)
    Wout = train(res_params, states, data)
    x = states[:, -1]
    return x, Wout, A, Win, states

def train(res_params, states, data):
    beta = res_params['beta']
    idenmat = beta * sparse.identity(res_params['D'])
    states2 = states.copy()

    #################################
    # Change the expansion function
    for j in range(1, np.shape(states2)[0] - 2):
        if (np.mod(j, 2) == 0):
            states2[j, :] = (states[j - 1, :] * states[j + 1, :]).copy()
    U = np.dot(states2, states2.transpose()) + idenmat
    Uinv = np.linalg.inv(U)
    Wout = np.dot(Uinv, np.dot(states2, data.transpose()))
    return Wout.transpose()

def predict(A, Win, res_params, x, Wout):
    output = np.zeros((res_params['num_inputs'], res_params['predict_length']))
    for i in range(res_params['predict_length']):
        x_aug = x.copy()
        ###############################
        for j in range(1, np.shape(x_aug)[0] - 2):
            if (np.mod(j, 2) == 0):
                x_aug[j] = (x[j - 1] * x[j + 1]).copy()
        out = np.squeeze(np.asarray(np.dot(Wout, x_aug)))
        output[:, i] = out
        x1 = np.tanh(np.dot(A, x) + np.dot(Win, out))
        x = np.squeeze(np.asarray(x1))
    return output, x

def calculate_ClustersSilhouetteScore(data,labels):
    # The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters.
    # Negative values generally indicate that a sample has been assigned to the wrong cluster.
    silhouette_average = silhouette_score(data, labels)
    return np.round(silhouette_average,4)

def run1():
    # This function is to draw the reservoir states as points in scattered plots.
    # It is used on the embedded matrix to draw the reduced states space over time in 2D to study the states clusters.
    # You can draw the states in steps instead of the whole range of time steps by controlling the step, min_rane and max_range variables

    # Variable used to enumerate figures
    num = 0
    for filename in os.listdir(path):
        embedding = np.load(path+filename)
        # Start time step index of the embedded matrix
        min_range = 0
        # End time step index of the embedded matrix
        max_range = 100000
        # Step to draw the states plot of the embedded matrix
        step = 1000
        # Get the minimum value of component 1 of embedding
        # -5 for better visualization
        min_x = np.min(embedding[:,0])-5
        # Get the maximum value of component 1 of embedding
        # +5 for better visualization
        max_x = np.max(embedding[:,0])+5
        # Get the minimum value of component 2 of embedding
        # -5 for better visualization
        min_y = np.min(embedding[:,1])-5
        # Get the maximum value of component 2 of embedding
        # +5 for better visualization
        max_y = np.max(embedding[:,1])+5
        for i in range(min_range, max_range, step):
            # To avoid overlapping plots in the same figure
            plt.figure(num)
            # c for the sequence of color specifications of each step
            # s is the marker (point) size on the plot, set s to 1
            plt.scatter(embedding[i:i+step, 0], embedding[i:i+step, 1],
                        c=range(i,i+step),s=1)
            # Labels for x-axis and y-axis of the plot
            plt.xlabel('First Component')
            plt.ylabel('Second Component')
            # Title of the plot named after the range of time steps
            plt.title(str(i/200)+" to " + str((i+step)/200) + " MTU")
            # Specifying the axes limits
            plt.xlim(min_x,max_x)
            plt.ylim(min_y, max_y)
            # Saving the plot in the same path as the code
            # Figure named after the range of time steps
            plt.savefig(path + str(i) +"_to_" + str(i+step) + ".png")
            plt.close()
            # Adding one to change the number of next figure
            num += 1

        num += 1
        plt.figure(num)
        plt.scatter(embedding[min_range:max_range, 0], embedding[min_range:max_range, 1],color='#0000FF' ,s=1 ,label='States trajectory')
        plt.scatter(embedding[0, 0],embedding[0, 1],marker='s',s=100,color='#00FF00',label='Start point of the trajectory')
        plt.scatter(embedding[embedding.shape[0]-1, 0], embedding[embedding.shape[0]-1, 1], marker='o', s=50, color='#FF0000', label='End point of the trajectory')
        plt.xlabel('First Component')
        plt.ylabel('Second Component')
        plt.title(str(embedding.shape))
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(path + filename + "_All.png")
        plt.close()

def run2():
    # This function applies KMeans with K = 5:20 and
    # Calculates the silhouette score for a given UMAP embedding and the KMeans labels after clustering
    # You can control the ranges of time steps to do the KMeans clustering by changing the values in List_range

    # Variable used to enumerate figures
    num = 0
    for filename in os.listdir(path):
        filehandle = open(path + filename + '_silhouette.txt', 'w')
        output = np.load(path+filename)
        List_range = ['0', '20000', '40000', '60000', '80000', '100000']
        filehandle.write("5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20"+"\n")
        for sel_range in range(0,len(List_range)-1):
            line = ""
            for clusters in range(5,21,1):
                min_range = int(List_range[sel_range])
                max_range = int(List_range[sel_range+1])
                currentData = output[min_range:max_range,:]
                kmeans = KMeans(n_clusters=clusters).fit(currentData)
                array = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}
                silhouette_average = calculate_ClustersSilhouetteScore(currentData,kmeans.labels_)
                line = line + str(silhouette_average) + ","
            filehandle.write(str(max_range)+","+line+"\n")
        filehandle.close()

def run3():
    # This function applies UMAP embedding on the states reservoir r(t) transpose of size (res_params['train_length'] X approx_res_size)
    # You can change the UMAP parameters by controlling n_components, n_neighbors, min_dist and metric.

    # train reservoir
    x, Wout, A, Win, states_res = train_reservoir(res_params,
                                                  measurements[:, shift_k:shift_k + res_params['train_length']])

    metric_options = ['correlation', 'cosine', 'euclidean']
    for i in range(35, 40, 5):  # Number of neighbors (7 settings)
         for j in range(3, 4):  # Enumeration over the min_dist values (3 settings)
             for k in range(2, 3): # Enumeration over the components values (1 setting)
                for m in metric_options:  # Enumeration over the metric options (3 settings)
                    embedding = umap.UMAP(n_components=k,n_neighbors=i,
                                           min_dist=j * 0.1,
                                           metric=m).fit_transform(np.transpose(states_res)) #np.transpose(states_res)

                    min_dist = round(j * 0.1, 1)
                    name = 'N=' + str(embedding.shape[0]) + '_' + 'Neighbors=' + str(i) + '_' + 'min_dist=' + str(
                            min_dist) + '_metric=' + m
                    np.save(path + name + '_'+str(k)+'_components.npy', embedding)

def run4():
    # This function trains the ESN and do 10 runs for every initial condition (10 runs X 100 initial conditions)
    # It saves the truth and prediction files for each of 1,000 experiments for further analysis
    # You can control the list and number of initial conditions by changing all_shifts

    # List of 100 Initial conditions
    all_shifts = []
    for i in range(0,900000,9150):
        all_shifts.append(i)
    all_shifts.append(898000)

    for shift in range(len(all_shifts)):
        # Reading the initial condition from the list
        shift_k = all_shifts[shift]
        # 10 runs
        for i in range(1,11):
            # train reservoir
            x, Wout, A, Win, states_res = train_reservoir_different_seed(res_params,
                                                      measurements[:, shift_k:shift_k + res_params['train_length']],i)

            truth = measurements[:,shift_k + res_params['train_length']:shift_k + res_params['train_length']+res_params['predict_length']]

            # prediction
            prediction, _ = predict(A, Win, res_params, x, Wout)

            np.save(path + str(shift_k) + '_truth_' + str(i)+'.npy', truth)
            np.save(path + str(shift_k) + '_prediction_' + str(i) + '.npy', prediction)

def run5():
    # This function uses the truth and prediction files generated by run4() and draw some analytical figures from them
    # It draws figures of the L2-norm for all 10 runs of each initial condition
    # It draws figures that show the prediction horizon of each initial condition
    # It draws figures that show the frequency distribution of the prediction horizons for all initial conditions
    # It draws figures that highlight the initial conditions with the best and worst prediction horizons

    # List of 100 Initial conditions
    all_shifts = []
    for i in range(0, 9500000, 102000):
        all_shifts.append(i)

    # Dictionary to store (Shift, MTU Prediction  Horizon)
    dict_ShifPredHorizon = {}

    for shift in range(len(all_shifts)):
        relative_l2_norm = []
        for i in range(1,11):
            # Reading the initial condition from the list
            shift_k = all_shifts[shift]
            truth_filename = path + str(shift_k) + "_truth_" + str(i)+".npy"
            prediction_filename = path + str(shift_k) + "_prediction_" + str(i)+".npy"
            truth = np.load(truth_filename)
            prediction = np.load(prediction_filename)
            plt.figure(figsize=(16, 32)).suptitle("Shift = " + str(shift_k) + "_Run " + str(i), fontsize=32)
            for k in range(0, 8):
                # Draw 8 subplots in the same figure
                plt.subplot(8, 1, k + 1)
                # Prediction is of (8,res_params['predict_length'])
                # Divide time by 200 to convert to MTU
                plt.plot((np.arange(0, res_params['predict_length']))/200, prediction[k, :], color='red', label='Pred')
                plt.plot((np.arange(0, res_params['predict_length']))/200, truth[k,:], color='blue', label='Truth')
                # Calculating MSE between truth and prediction
                rmse = np.sqrt((np.square(truth[k, :] - prediction[k, :])).mean(axis=None))
                # Label y axis and put MSE as part of it
                plt.ylabel('X' + str(k) + "\n" + "RMSE = " + str(np.round(rmse,4)))
                plt.xlabel('MTU')
                # Set the range of values in the x-axis
                plt.xlim(0,10)
                # Set the tick marks (labels) of the x-axis
                plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
                plt.legend(loc='upper right')

            plt.savefig(path + "shift_"+str(shift_k) + "_run_" + str(i) + ".png")
            plt.close()

            relative_l2_norm.append(np.linalg.norm(truth - prediction,axis=0)/(np.linalg.norm(truth,axis=0).mean(axis=0)))

        # L-2 norm Figure (Line for each run)
        plt.figure(11)
        for i in range(1,11):
            plt.plot((np.arange(0, res_params['predict_length'])) / 200, relative_l2_norm[i-1], label='Run'+str(i))

        plt.ylabel('L2-norm(t)')
        plt.xlabel('Prediction Horizon in MTU')
        # Set the range of values in the x-axis
        plt.xlim(0, 10)
        # Set the tick marks (labels) of both axes
        plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],rotation = "vertical")
        plt.yticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
        plt.legend(loc='upper left')
        plt.title("Shift = " + str(shift_k) + " L2-norm for all 10 Runs")
        plt.tight_layout()
        plt.savefig(path + str(shift_k) + "_L2-norm.png")
        plt.close()

        # Prediction Horizon Figure

        # Converting relative l2 norm list to an array
        array_l2_norm = np.array(relative_l2_norm)
        # Getting mean of relative l2 norms over all 10 runs
        mean_l2_norm = np.mean(array_l2_norm,axis=0)
        # Getting standard deviation of relative l2 norms over all 10 runs
        std_l2_norm = np.std(array_l2_norm, axis=0)

        # Getting the MTU value corresponding to 0.3 relative l2 norm
        l2_norm_threshold = 0.3
        for MTU_index in range(0,res_params['predict_length']):
            if(mean_l2_norm[MTU_index] > l2_norm_threshold):
                MTU_index = MTU_index-1
                break

        plt.figure(12)
        plt.plot((np.arange(0, res_params['predict_length'])) / 200, mean_l2_norm+std_l2_norm, color ='#0000FF', label='10 Runs Mean+STD')
        plt.plot((np.arange(0, res_params['predict_length'])) / 200, mean_l2_norm, color ='#FF0000',label='10 Runs Mean')
        plt.plot((np.arange(0, res_params['predict_length'])) / 200, mean_l2_norm-std_l2_norm, color = '#008000', label='10 Runs Mean-STD')

        plt.axhline(y=l2_norm_threshold,color = '#000000',linestyle='--', label='L2-norm Threshold')
        plt.axvline(x=np.round(MTU_index/200,1),color = '#000000',linestyle=':', label = 'Prediction Horizon Threshold')

        plt.ylabel('L2-norm(t)')
        plt.xlabel('Prediction Horizon in MTU')
        # Set the range of values in the x-axis
        plt.xlim(0, 10)
        # Set the tick marks (labels) of the x-axis
        plt.xticks([0,5,10,np.round(MTU_index/200,1)],rotation = "vertical")
        plt.yticks([0,0.5,1,1.5,2,2.5,3,l2_norm_threshold])
        plt.legend(loc='upper right')
        plt.title("Shift = " + str(shift_k) + " Prediction Horizon")
        plt.tight_layout()
        plt.savefig(path + str(shift_k) + "_PredictionHorizon.png")
        plt.close()

        dict_ShifPredHorizon.update({str(shift_k):np.round(MTU_index/200,1)})

    int_docs_info = {int(k): v for k, v in dict_ShifPredHorizon.items()}
    sorted_dict = OrderedDict(sorted(int_docs_info.items()))  # sorted by key, return a list of tuples

    # Figure to show The Frequency Distribution of Prediction Horizons of 100 ICs
    fig = plt.figure(13)
    ax = fig.add_subplot(111)
    freq, _, _ = ax.hist(sorted_dict.values(), 100, color='#000000', label='Frequency')
    mean_MTU = np.mean(list(sorted_dict.values()))
    std_MTU = np.std(list(sorted_dict.values()))

    ax.axvline(x=mean_MTU+std_MTU, color='#0000FF',linewidth=2,label='Mean+/-STD',linestyle=':')
    ax.axvline(x=mean_MTU,color='#FF0000',linewidth=2,label='Mean',linestyle='--')
    ax.axvline(x=mean_MTU-std_MTU, color='#0000FF',linewidth=2, linestyle=':')
    ax.axvspan(mean_MTU+std_MTU, max(sorted_dict.values()), alpha=0.2, color='#006400',label ='Easy ICs')
    ax.axvspan(min(sorted_dict.values()),mean_MTU-std_MTU, alpha=0.2, color='#8B0000', label='Hard ICs')
    ax.set_xlim(0, max(sorted_dict.values())+1)
    ax.set_ylim(0, max(freq) + 1)
    plt.xticks([0,5,np.round(mean_MTU,1),np.round(mean_MTU+std_MTU,1),np.round(mean_MTU-std_MTU,1),np.round(max(sorted_dict.values()),1),np.round(min(sorted_dict.values()),1)], rotation="vertical")
    ax.set_xlabel('Prediction Horizon in MTU')
    ax.set_ylabel('Frequency')
    plt.legend(loc='upper right')
    ax.set_title('Frequency Distribution of Prediction Horizons (100 ICs)')
    fig.tight_layout()

    fig.savefig(path + "FrequencyDistributionPredictionHorizons_100ICs.png")
    plt.close()

    # Figure to show The Prediction Horizon of 100 Initial Conditions
    east_ICs = {}
    hard_ICs= {}
    for k,v in  sorted_dict.items():
        if v>= mean_MTU+std_MTU and v<= max(sorted_dict.values()):
            east_ICs.update({k: v})

    sorted_east_ICs = OrderedDict(sorted(east_ICs.items()))  # sorted by key, return a list of tuples

    for k,v in  sorted_dict.items():
        if v>= min(sorted_dict.values()) and v <= mean_MTU-std_MTU:
            hard_ICs.update({k: v})
    sorted_hard_ICs = OrderedDict(sorted(hard_ICs.items()))  # sorted by key, return a list of tuples

    plt.figure(14)
    fig, ax1 = plt.subplots()
    ax1.scatter(sorted_east_ICs.keys(), sorted_east_ICs.values(), color='#006400',marker='s',label='Easy IC')
    ax1.scatter(sorted_hard_ICs.keys(), sorted_hard_ICs.values(), color='#FF0000',marker='v', label='Hard IC')
    ax1.plot(sorted_dict.keys(),sorted_dict.values(),color='#0000FF', label ='Prediction Horizon of Initial Condition')
    ax1.axhline(y=max(sorted_dict.values()), color='#006400', linestyle='-.', label='Highest & Lowest Prediction Horizons')
    ax1.axhline(y=min(sorted_dict.values()), color='#FF0000', linestyle='-.')
    ax1.axhline(y=np.mean(list(sorted_dict.values())), color='#000000', linestyle='--', label='Mean of Prediction Horizons')
    ax1.axhline(y=np.round(np.mean(list(sorted_dict.values()))+np.std(list(sorted_dict.values())),1), color='#006400', linestyle=':',label='Mean+/-STD')
    ax1.axhline(y=np.round(np.mean(list(sorted_dict.values()))-np.std(list(sorted_dict.values())),1),color='#FF0000', linestyle=':')

    ax1.set_ylabel('Prediction Horizon in MTU')
    ax1.set_xlabel('Initial Condition')
    ax1.set_ylim(0,max(sorted_dict.values())+3)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylim(0, max(sorted_dict.values())+3)
    ax2.set_yticks([max(sorted_dict.values()),min(sorted_dict.values()),np.round(np.mean(list(sorted_dict.values())),1),
                    np.round(np.mean(list(sorted_dict.values())) + np.std(list(sorted_dict.values())), 1),
                    np.round(np.mean(list(sorted_dict.values())) - np.std(list(sorted_dict.values())), 1)])
    ax2.set_ylabel('Prediction Horizons Statistics in MTU')

    ax1.legend(loc='upper right')
    plt.title("Prediction Horizon (100 ICs)")
    fig.tight_layout()
    plt.savefig(path + "PredictionHorizon_100ICs.png")
    plt.close()

def main():
    # The main function to control which function to run when called from the command line
    run1()

if __name__ == '__main__':
    main()
