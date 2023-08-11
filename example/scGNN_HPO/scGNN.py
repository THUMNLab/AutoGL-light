import time
import os
import argparse
import sys
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import resource
import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.nn import functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans, SpectralClustering, AffinityPropagation, AgglomerativeClustering, Birch, DBSCAN, FeatureAgglomeration, OPTICS, MeanShift
from model import AE, VAE
from util_function import *
from graph_function import *
from benchmark_util import *
from gae_embedding import GAEembedding, measure_clustering_results, test_clustering_benchmark_results
import torch.multiprocessing as mp
from autogllight.hpo import build_hpo_from_name, LogRangeHP, ChoiceHP

#torch.cuda.set_device(1)
#os.environ['CUDA_VISIBLE_DEVICES'] = "1"

def get_parser():
    PARSER = argparse.ArgumentParser(description='Main entrance of scGNN')

    PARSER.add_argument('--datasetName', type=str, default='GSE138852', ##################'481193cb-c021-4e04-b477-0b7cfef4614b.mtx',
                        help='For 10X: folder name of 10X dataset; For CSV: csv file name')
    PARSER.add_argument('--datasetDir', type=str, default='./',  ####################'/storage/htc/joshilab/wangjue/casestudy/',
                        help='Directory of dataset: default(/home/wangjue/biodata/scData/10x/6/)')

    PARSER.add_argument('--batch-size', type=int, default=12800, metavar='N',
                        help='input batch size for training (default: 12800)')
    PARSER.add_argument('--Regu-epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train in Feature Autoencoder initially (default: 500)')
    PARSER.add_argument('--EM-epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train Feature Autoencoder in iteration EM (default: 200)')
    PARSER.add_argument('--EM-iteration', type=int, default=10, metavar='N',
                        help='number of iteration in total EM iteration (default: 10)')
    PARSER.add_argument('--quickmode', action='store_true', default=True, ##################False
                        help='whether use quickmode, skip Cluster Autoencoder (default: no quickmode)')
    PARSER.add_argument('--cluster-epochs', type=int, default=200, metavar='N',
                        help='number of epochs in Cluster Autoencoder training (default: 200)')
    PARSER.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable GPU training. If you only have CPU, add --no-cuda in the command line')
    PARSER.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    PARSER.add_argument('--regulized-type', type=str, default='noregu',
                        help='regulized type (default: LTMG) in EM, otherwise: noregu/LTMG/LTMG01')
    PARSER.add_argument('--reduction', type=str, default='sum',
                        help='reduction type: mean/sum, default(sum)')
    PARSER.add_argument('--model', type=str, default='AE',
                        help='VAE/AE (default: AE)')
    PARSER.add_argument('--gammaPara', type=float, default=0.1,
                        help='regulized intensity (default: 0.1)')
    PARSER.add_argument('--alphaRegularizePara', type=float, default=0.9,
                        help='regulized parameter (default: 0.9)')

    # Build cell graph
    PARSER.add_argument('--k', type=int, default=10,
                        help='parameter k in KNN graph (default: 10)')
    PARSER.add_argument('--knn-distance', type=str, default='euclidean',
                        help='KNN graph distance type: euclidean/cosine/correlation (default: euclidean)')
    PARSER.add_argument('--prunetype', type=str, default='KNNgraphStatsSingleThread',
                        help='prune type, KNNgraphStats/KNNgraphML/KNNgraphStatsSingleThread (default: KNNgraphStatsSingleThread)')

    # Debug related
    PARSER.add_argument('--precisionModel', type=str, default='Float',
                        help='Single Precision/Double precision: Float/Double (default:Float)')
    PARSER.add_argument('--coresUsage', type=str, default='1',
                        help='how many cores used: all/1/... (default:1)')
    PARSER.add_argument('--outputDir', type=str, default='outputDir/',
                        help='save npy results in directory')
    PARSER.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    PARSER.add_argument('--saveinternal', action='store_true', default=False,
                        help='whether save internal interation results or not')
    PARSER.add_argument('--debugMode', type=str, default='noDebug',
                        help='savePrune/loadPrune for extremely huge data in debug (default: noDebug)')
    PARSER.add_argument('--nonsparseMode', action='store_true', default=True, #################False,
                        help='SparseMode for running for huge dataset')

    # LTMG related
    PARSER.add_argument('--LTMGDir', type=str, default='/storage/htc/joshilab/wangjue/casestudy/',
                        help='directory of LTMGDir, default:(/home/wangjue/biodata/scData/allBench/)')
    PARSER.add_argument('--ltmgExpressionFile', type=str, default='Use_expression.csv',
                        help='expression File after ltmg in csv')
    PARSER.add_argument('--ltmgFile', type=str, default='LTMG_sparse.mtx',
                        help='expression File in csv. (default:LTMG_sparse.mtx for sparse mode/ ltmg.csv for nonsparse mode) ')

    # Clustering related
    PARSER.add_argument('--useGAEembedding', action='store_true', default=False,
                        help='whether use GAE embedding for clustering(default: False)')
    PARSER.add_argument('--useBothembedding', action='store_true', default=False,
                        help='whether use both embedding and Graph embedding for clustering(default: False)')
    PARSER.add_argument('--n-clusters', default=20, type=int,
                        help='number of clusters if predifined for KMeans/Birch ')
    PARSER.add_argument('--clustering-method', type=str, default='LouvainK',
                        help='Clustering method: Louvain/KMeans/SpectralClustering/AffinityPropagation/AgglomerativeClustering/AgglomerativeClusteringK/Birch/BirchN/MeanShift/OPTICS/LouvainK/LouvainB')
    PARSER.add_argument('--maxClusterNumber', type=int, default=30,
                        help='max cluster for celltypeEM without setting number of clusters (default: 30)')
    PARSER.add_argument('--minMemberinCluster', type=int, default=5,
                        help='max cluster for celltypeEM without setting number of clusters (default: 100)')
    PARSER.add_argument('--resolution', type=str, default='auto',
                        help='the number of resolution on Louvain (default: auto/0.5/0.8)')

    # imputation related
    PARSER.add_argument('--EMregulized-type', type=str, default='Celltype',
                        help='regulized type (default: noregu) in EM, otherwise: noregu/Graph/GraphR/Celltype')
    PARSER.add_argument('--gammaImputePara', type=float, default=0.0,
                        help='regulized parameter (default: 0.0)')
    PARSER.add_argument('--graphImputePara', type=float, default=0.3,
                        help='graph parameter (default: 0.3)')
    PARSER.add_argument('--celltypeImputePara', type=float, default=0.1,
                        help='celltype parameter (default: 0.1)')
    PARSER.add_argument('--L1Para', type=float, default=1.0,
                        help='L1 regulized parameter (default: 0.001)')
    PARSER.add_argument('--L2Para', type=float, default=0.0,
                        help='L2 regulized parameter (default: 0.001)')
    PARSER.add_argument('--EMreguTag', action='store_true', default=False,
                        help='whether regu in EM process')
    PARSER.add_argument('--sparseImputation', type=str, default='nonsparse',
                        help='whether use sparse in imputation: sparse/nonsparse (default: nonsparse)')

    # dealing with zeros in imputation results
    PARSER.add_argument('--zerofillFlag', action='store_true', default=False,
                        help='fill zero or not before EM process (default: False)')
    PARSER.add_argument('--noPostprocessingTag', action='store_false', default=True,
                        help='whether postprocess imputated results, default: (True)')
    PARSER.add_argument('--postThreshold', type=float, default=0.01,
                        help='Threshold to force expression as 0, default:(0.01)')

    # Converge related
    PARSER.add_argument('--alpha', type=float, default=0.5,
                        help='iteration alpha (default: 0.5) to control the converge rate, should be a number between 0~1')
    PARSER.add_argument('--converge-type', type=str, default='celltype',
                        help='type of converge condition: celltype/graph/both/either (default: celltype) ')
    PARSER.add_argument('--converge-graphratio', type=float, default=0.01,
                        help='converge condition: ratio of graph ratio change in EM iteration (default: 0.01), 0-1')
    PARSER.add_argument('--converge-celltyperatio', type=float, default=0.99,
                        help='converge condition: ratio of cell type change in EM iteration (default: 0.99), 0-1')

    # GAE related
    PARSER.add_argument('--GAEmodel', type=str,
                        default='gcn_vae', help="models used")
    PARSER.add_argument('--GAEepochs', type=int, default=200,
                        help='Number of epochs to train.')
    PARSER.add_argument('--GAEhidden1', type=int, default=32,
                        help='Number of units in hidden layer 1.')
    PARSER.add_argument('--GAEhidden2', type=int, default=16,
                        help='Number of units in hidden layer 2.')
    PARSER.add_argument('--GAElr', type=float, default=0.01,
                        help='Initial learning rate.')
    PARSER.add_argument('--GAEdropout', type=float, default=0.,
                        help='Dropout rate (1 - keep probability).')
    PARSER.add_argument('--GAElr_dw', type=float, default=0.001,
                                help='Initial learning rate for regularization.')
    
    return PARSER


class scGNN:
    def __init__(self, PARSER):
        self.args = PARSER.parse_args()
        self.args.cuda = not self.args.no_cuda and torch.cuda.is_available()
        self.args.sparseMode = not self.args.nonsparseMode

        checkargs(self.args)

        torch.manual_seed(self.args.seed)
        self.device = torch.device("cuda" if self.args.cuda else "cpu")
        print('Using device:'+str(self.device))

        if not self.args.coresUsage == 'all':
            torch.set_num_threads(int(self.args.coresUsage))

        self.kwargs = {'num_workers': 0, 'pin_memory': True} if self.args.cuda else {}
        # print(args)
        self.start_time = time.time()

        # load scRNA in csv
        print('---0:00:00---scRNA starts loading.')
        self.data, self.genelist, self.celllist = loadscExpression(
            self.args.datasetDir+self.args.datasetName+'/'+self.args.ltmgExpressionFile, sparseMode=self.args.sparseMode)
        print('---'+str(datetime.timedelta(seconds=int(time.time()-self.start_time))) +
            '---scRNA has been successfully loaded')

        self.scData = scDataset(self.data)
        self.train_loader = DataLoader(
            self.scData, batch_size=self.args.batch_size, shuffle=False, **self.kwargs)
        print('---'+str(datetime.timedelta(seconds=int(time.time()-self.start_time))) +
            '---TrainLoader has been successfully prepared.')

        # load LTMG in sparse version
        if not self.args.regulized_type == 'noregu':
            print('Start loading LTMG in sparse coding.')
            self.regulationMatrix = readLTMG(
                self.args.LTMGDir+self.args.datasetName+'/', self.args.ltmgFile)
            self.regulationMatrix = torch.from_numpy(self.regulationMatrix)
            if self.args.precisionModel == 'Double':
                self.regulationMatrix = self.regulationMatrix.type(torch.DoubleTensor)
            elif self.args.precisionModel == 'Float':
                self.regulationMatrix = self.regulationMatrix.type(torch.FloatTensor)
            print('---'+str(datetime.timedelta(seconds=int(time.time()-self.start_time))
                            )+'---LTMG has been successfully prepared.')
        else:
            self.regulationMatrix = None

    def get_model(self, hidden_dim = 128, hidden_length = 0):
        # Original
        if self.args.model == 'VAE':
            self.model = VAE(dim=self.scData.features.shape[1], hidden_dim=hidden_dim, hidden_length=hidden_length).to(self.device)
        elif self.args.model == 'AE':
            self.model = AE(dim=self.scData.features.shape[1], hidden_dim=hidden_dim, hidden_length=hidden_length).to(self.device)
        if self.args.precisionModel == 'Double':
            self.model = self.model.double()

    def get_optimizer(self, lr = 1e-3, weight_decay = 1e-4):
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay = weight_decay)
        print('---'+str(datetime.timedelta(seconds=int(time.time()-self.start_time))) +
            '---Pytorch model ready.')

    def train_inside(self, epoch, train_loader=None, EMFlag=False, taskType='celltype', sparseImputation='nonsparse'):
        '''
        EMFlag indicates whether in EM processes. 
            If in EM, use regulized-type parsed from program entrance,
            Otherwise, noregu
            taskType: celltype or imputation
        '''

        self.model.train()
        train_loss = 0
        for batch_idx, (data, dataindex) in enumerate(train_loader):
            if self.args.precisionModel == 'Double':
                data = data.type(torch.DoubleTensor)
            elif self.args.precisionModel == 'Float':
                data = data.type(torch.FloatTensor)
            data = data.to(self.device)
            if not self.args.regulized_type == 'noregu':
                regulationMatrixBatch = self.regulationMatrix[dataindex, :]
                regulationMatrixBatch = regulationMatrixBatch.to(self.device)
            else:
                regulationMatrixBatch = None
            if taskType == 'imputation':
                if sparseImputation == 'nonsparse':
                    celltypesampleBatch = self.celltypesample[dataindex,
                                                        :][:, dataindex]
                    adjsampleBatch = self.adjsample[dataindex, :][:, dataindex]
                elif sparseImputation == 'sparse':
                    celltypesampleBatch = generateCelltypeRegu(
                        self.listResult[dataindex])
                    celltypesampleBatch = torch.from_numpy(celltypesampleBatch)
                    if self.args.precisionModel == 'Float':
                        celltypesampleBatch = celltypesampleBatch.float()
                    elif self.args.precisionModel == 'Double':
                        celltypesampleBatch = celltypesampleBatch.type(
                            torch.DoubleTensor)
                    # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    # print('celltype Mem consumption: '+str(mem))

                    adjsampleBatch = self.adj[dataindex, :][:, dataindex]
                    adjsampleBatch = sp.csr_matrix.todense(adjsampleBatch)
                    adjsampleBatch = torch.from_numpy(adjsampleBatch)
                    if self.args.precisionModel == 'Float':
                        adjsampleBatch = adjsampleBatch.float()
                    elif self.args.precisionModel == 'Double':
                        adjsampleBatch = adjsampleBatch.type(torch.DoubleTensor)
                    # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    # print('adj Mem consumption: '+str(mem))

            self.optimizer.zero_grad()
            if self.args.model == 'VAE':
                recon_batch, mu, logvar, z = self.model(data)
                if taskType == 'celltype':
                    if EMFlag and (not self.args.EMreguTag):
                        loss = loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu, logvar, gammaPara=self.args.gammaPara, regulationMatrix=regulationMatrixBatch,
                                                regularizer_type='noregu', reguPara=self.args.alphaRegularizePara, modelusage=self.args.model, reduction=self.args.reduction)
                    else:
                        loss = loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu, logvar, gammaPara=self.args.gammaPara, regulationMatrix=regulationMatrixBatch,
                                                regularizer_type=self.args.regulized_type, reguPara=self.args.alphaRegularizePara, modelusage=self.args.model, reduction=self.args.reduction)
                elif taskType == 'imputation':
                    if EMFlag and (not self.args.EMreguTag):
                        loss = loss_function_graph_celltype(recon_batch, data.view(-1, recon_batch.shape[1]), mu, logvar, graphregu=adjsampleBatch, celltyperegu=celltypesampleBatch, gammaPara=self.args.gammaImputePara,
                                                            regulationMatrix=regulationMatrixBatch, regularizer_type=self.args.EMregulized_type, reguPara=self.args.graphImputePara, reguParaCelltype=self.args.celltypeImputePara, modelusage=self.args.model, reduction=self.args.reduction)
                    else:
                        loss = loss_function_graph_celltype(recon_batch, data.view(-1, recon_batch.shape[1]), mu, logvar, graphregu=adjsampleBatch, celltyperegu=celltypesampleBatch, gammaPara=self.args.gammaImputePara,
                                                            regulationMatrix=regulationMatrixBatch, regularizer_type=self.args.regulized_type, reguPara=self.args.graphImputePara, reguParaCelltype=self.args.celltypeImputePara, modelusage=self.args.model, reduction=self.args.reduction)

            elif self.args.model == 'AE':
                recon_batch, z = self.model(data)
                mu_dummy = ''
                logvar_dummy = ''
                if taskType == 'celltype':
                    if EMFlag and (not self.args.EMreguTag):
                        loss = loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu_dummy, logvar_dummy, gammaPara=self.args.gammaPara,
                                                regulationMatrix=regulationMatrixBatch, regularizer_type='noregu', reguPara=self.args.alphaRegularizePara, modelusage=self.args.model, reduction=self.args.reduction)
                    else:
                        loss = loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu_dummy, logvar_dummy, gammaPara=self.args.gammaPara, regulationMatrix=regulationMatrixBatch,
                                                regularizer_type=self.args.regulized_type, reguPara=self.args.alphaRegularizePara, modelusage=self.args.model, reduction=self.args.reduction)
                elif taskType == 'imputation':
                    if EMFlag and (not self.args.EMreguTag):
                        loss = loss_function_graph_celltype(recon_batch, data.view(-1, recon_batch.shape[1]), mu_dummy, logvar_dummy, graphregu=adjsampleBatch, celltyperegu=celltypesampleBatch, gammaPara=self.args.gammaImputePara,
                                                            regulationMatrix=regulationMatrixBatch, regularizer_type=self.args.EMregulized_type, reguPara=self.args.graphImputePara, reguParaCelltype=self.args.celltypeImputePara, modelusage=self.args.model, reduction=self.args.reduction)
                    else:
                        loss = loss_function_graph_celltype(recon_batch, data.view(-1, recon_batch.shape[1]), mu_dummy, logvar_dummy, graphregu=adjsampleBatch, celltyperegu=celltypesampleBatch, gammaPara=self.args.gammaImputePara,
                                                            regulationMatrix=regulationMatrixBatch, regularizer_type=self.args.regulized_type, reguPara=self.args.graphImputePara, reguParaCelltype=self.args.celltypeImputePara, modelusage=self.args.model, reduction=self.args.reduction)

            # L1 and L2 regularization
            # 0.0 for no regularization
            l1 = 0.0
            l2 = 0.0
            for p in self.model.parameters():
                l1 = l1 + p.abs().sum()
                l2 = l2 + p.pow(2).sum()
            loss = loss + self.args.L1Para * l1 + self.args.L2Para * l2

            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(data)))

            # for batch
            if batch_idx == 0:
                recon_batch_all = recon_batch
                data_all = data
                z_all = z
            else:
                recon_batch_all = torch.cat((recon_batch_all, recon_batch), 0)
                data_all = torch.cat((data_all, data), 0)
                z_all = torch.cat((z_all, z), 0)

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))

        return recon_batch_all, data_all, z_all, (train_loss / len(train_loader.dataset))


    def origin_main(self):
        self.start_time = time.time()
        self.adjsample = None
        self.celltypesample = None
        # If not exist, then create the outputDir
        if not os.path.exists(self.args.outputDir):
            os.makedirs(self.args.outputDir)
        # outParaTag = str(args.gammaImputePara)+'-'+str(args.graphImputePara)+'-'+str(args.celltypeImputePara)
        self.ptfileStart = self.args.outputDir+self.args.datasetName+'_EMtrainingStart.pt'
        # ptfile      = args.outputDir+args.datasetName+'_EMtraining.pt'


        # Debug
        if self.args.debugMode == 'savePrune' or self.args.debugMode == 'noDebug':
            # store parameter
            self.stateStart = {
                # 'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
            torch.save(self.stateStart, self.ptfileStart)
            print('Start training...')
            for epoch in range(1, self.args.Regu_epochs + 1):
                self.recon, self.original, self.z, self.Regu_loss = self.train_inside(epoch, EMFlag=False, train_loader=self.train_loader)

            self.zOut = self.z.detach().cpu().numpy()
            print('zOut ready at ' + str(time.time()-self.start_time))
            self.ptstatus = self.model.state_dict()

            # Store reconOri for imputation
            self.reconOri = self.recon.clone()
            self.reconOri = self.reconOri.detach().cpu().numpy()

            # Step 1. Inferring celltype

            # Here para = 'euclidean:10'
            # adj, edgeList = generateAdj(zOut, graphType='KNNgraphML', para = args.knn_distance+':'+str(args.k))
            print('---'+str(datetime.timedelta(seconds=int(time.time()-self.start_time)))+'---Start Prune')
            self.adj, self.edgeList = generateAdj(self.zOut, graphType=self.args.prunetype, para=self.args.knn_distance+':'+str(
                self.args.k), adjTag=(self.args.useGAEembedding or self.args.useBothembedding))
            print('---'+str(datetime.timedelta(seconds=int(time.time() -
                                                        self.start_time)))+'---Prune Finished')
            # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        
        # For iteration studies
        self.G0 = nx.Graph()
        self.G0.add_weighted_edges_from(self.edgeList)
        self.nlG0 = nx.normalized_laplacian_matrix(self.G0)
        # set iteration criteria for converge
        self.adjOld = self.nlG0
        # set celltype criteria for converge
        self.listResultOld = [1 for i in range(self.zOut.shape[0])]

        # Define resolution
        # Default: auto, otherwise use user defined resolution
        if self.args.resolution == 'auto':
            if self.zOut.shape[0] < 2000:
                self.resolution = 0.8
            else:
                self.resolution = 0.5
        else:
            self.resolution = float(self.args.resolution)

        print('---'+str(datetime.timedelta(seconds=int(time.time()-self.start_time))
                        )+"---EM process starts")

        for bigepoch in range(0, self.args.EM_iteration):
            print('---'+str(datetime.timedelta(seconds=int(time.time() -
                                                        self.start_time)))+'---Start %sth iteration.' % (bigepoch))

            # Now for both methods, we need do clustering, using clustering results to check converge
            # Clustering: Get clusters
            if self.args.clustering_method == 'LouvainK':
                self.listResult, size = generateLouvainCluster(self.edgeList)
                self.k = len(np.unique(self.listResult))
                print('Louvain cluster: '+str(self.k))
                self.k = int(self.k*self.resolution) if int(self.k*self.resolution)>=3 else 2
                self.clustering = KMeans(n_clusters=self.k, random_state=0).fit(self.zOut)
                self.listResult = self.clustering.predict(self.zOut)
           
            print('---'+str(datetime.timedelta(seconds=int(time.time() -
                                                        self.start_time)))+"---Clustering Ends")

            # If clusters more than maxclusters, then have to stop
            if len(set(self.listResult)) > self.args.maxClusterNumber or len(set(self.listResult)) <= 1:
                print("Stopping: Number of clusters is " +
                    str(len(set(self.listResult))) + ".")
                # Exit
                # return None
                # Else: dealing with the number
                self.listResult = trimClustering(
                    self.listResult, minMemberinCluster=self.args.minMemberinCluster, maxClusterNumber=self.args.maxClusterNumber)

            # Debug: Calculate silhouette
            # measure_clustering_results(zOut, listResult)
            print('Total Cluster Number: '+str(len(set(self.listResult))))
            # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # print('Mem consumption: '+str(mem))

            # Graph regulizated EM AE with Cluster AE, do the additional AE


            ################################## quickmode


            # Use new dataloader
            self.scDataInter = scDatasetInter(self.recon.detach().cpu())
            self.train_loader2 = DataLoader(
                self.scDataInter, batch_size=self.args.batch_size, shuffle=False, **self.kwargs)

            for epoch in range(1, self.args.EM_epochs + 1):
                self.recon, self.original, self.z, self.EM_loss = self.train_inside(epoch, EMFlag=True, train_loader=self.train_loader2)
                # recon, original, z = train(epoch, train_loader=train_loader, EMFlag=True)

            self.zOut = self.z.detach().cpu().numpy()

            # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # print('Mem consumption: '+str(mem))
            print('---'+str(datetime.timedelta(seconds=int(time.time()-self.start_time)))+'---Start Prune')
            self.adj, self.edgeList = generateAdj(self.zOut, graphType=self.args.prunetype, para=self.args.knn_distance+':'+str(
                self.args.k), adjTag=(self.args.useGAEembedding or self.args.useBothembedding or (bigepoch == int(self.args.EM_iteration)-1)))
            print('---'+str(datetime.timedelta(seconds=int(time.time() -
                                                        self.start_time)))+'---Prune Finished')
            # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # print('Mem consumption: '+str(mem))


            # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # print('Mem consumption: '+str(mem))
            print('---'+str(datetime.timedelta(seconds=int(time.time() -
                                                        self.start_time)))+'---Start test converge condition')

            # Iteration usage
            # If not only use 'celltype', we have to use graph change
            # The problem is it will consume huge memory for giant graphs


            # Check similarity
            self.ari = adjusted_rand_score(self.listResultOld, self.listResult)

            # Debug Information of clustering results between iterations
            # print(listResultOld)
            # print(listResult)
            print('celltype similarity:'+str(self.ari))

            # graph criteria
            if self.args.converge_type == 'celltype':
                if self.ari > self.args.converge_celltyperatio:
                    print('Converge now!')
                    break

            # Update
            self.listResultOld = self.listResult
            print('---'+str(datetime.timedelta(seconds=int(time.time()-self.start_time))
                            )+"---"+str(bigepoch)+"th iteration in EM Finished")

        # Use new dataloader
        print('---'+str(datetime.timedelta(seconds=int(time.time()-self.start_time))
                        )+"---Starts Imputation")
        # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # print('Mem consumption: '+str(mem))
        self.scDataInter = scDatasetInter(self.reconOri)
        self.train_loader = DataLoader(
            self.scDataInter, batch_size=self.args.batch_size, shuffle=False, **self.kwargs)

        self.stateStart = torch.load(self.ptfileStart)
        self.model.load_state_dict(self.stateStart['state_dict'])
        self.optimizer.load_state_dict(self.stateStart['optimizer'])
        # model.load_state_dict(torch.load(ptfileStart))
        # if args.aePara == 'start':
        #     model.load_state_dict(torch.load(ptfileStart))
        # elif args.aePara == 'end':
        #     model.load_state_dict(torch.load(ptfileEnd))

        # generate graph regularizer from graph
        # adj = adj.tolist() # Used for read/load
        # adjdense = sp.csr_matrix.todense(adj)

        # Better option: use torch.sparse
        if self.args.sparseImputation == 'nonsparse':
            # generate adj from edgeList
            self.adjdense = sp.csr_matrix.todense(self.adj)
            self.adjsample = torch.from_numpy(self.adjdense)
            if self.args.precisionModel == 'Float':
                self.adjsample = self.adjsample.float()
            elif self.args.precisionModel == 'Double':
                self.adjsample = self.adjsample.type(torch.DoubleTensor)
            self.adjsample = self.adjsample.to(self.device)
            # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # print('Mem consumption: '+str(mem))

            # generate celltype regularizer from celltype
            self.celltypesample = generateCelltypeRegu(self.listResult)
            self.celltypesample = torch.from_numpy(self.celltypesample)
            if self.args.precisionModel == 'Float':
                self.celltypesample = self.celltypesample.float()
            elif self.args.precisionModel == 'Double':
                self.celltypesample = self.celltypesample.type(torch.DoubleTensor)
            self.celltypesample = self.celltypesample.to(self.device)
            # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # print('Mem consumption: '+str(mem))

        for epoch in range(1, self.args.EM_epochs + 1):
            self.recon, self.original, self.z, self.last_loss = self.train_inside(
                epoch, EMFlag=True, taskType='imputation', sparseImputation=self.args.sparseImputation,
                train_loader=self.train_loader)

        self.reconOut = self.recon.detach().cpu().numpy()
        if not self.args.noPostprocessingTag:
            self.threshold_indices = self.reconOut < self.args.postThreshold
            self.reconOut[self.threshold_indices] = 0.0

        # Output final results
        print('---'+str(datetime.timedelta(seconds=int(time.time()-self.start_time))
                        )+'---All iterations finished, start output results.')
        # Output imputation Results
        # np.save   (args.npyDir+args.datasetName+'_'+args.regulized_type+'_'+outParaTag+'_recon.npy',reconOut)
        # np.savetxt(args.npyDir+args.datasetName+'_'+args.regulized_type+'_'+outParaTag+'_recon.csv',reconOut,delimiter=",",fmt='%10.4f')
        # Output celltype Results

        self.all_lost = self.EM_loss + self.Regu_loss + self.last_loss

        self.recon_df = pd.DataFrame(np.transpose(self.reconOut),
                                index=self.genelist, columns=self.celllist)
        self.recon_df.to_csv(self.args.outputDir+self.args.datasetName+'_recon.csv')
        self.emblist = []
        for i in range(self.zOut.shape[1]):
            self.emblist.append('embedding'+str(i))
        self.embedding_df = pd.DataFrame(self.zOut, index=self.celllist, columns=self.emblist)
        self.embedding_df.to_csv(self.args.outputDir+self.args.datasetName+'_'+str(self.all_lost)+'_embedding.csv')
        self.graph_df = pd.DataFrame(self.edgeList, columns=["NodeA", "NodeB", "Weights"])
        self.graph_df.to_csv(self.args.outputDir+self.args.datasetName+'_'+str(self.all_lost)+'_graph.csv', index=False)
        self.results_df = pd.DataFrame(self.listResult, index=self.celllist, columns=["Celltype"])
        self.results_df.to_csv(self.args.outputDir+self.args.datasetName+'_'+str(self.all_lost)+'_results.txt')

        print(f"############LOSS:{self.EM_loss}+{self.Regu_loss}+{self.last_loss} = {self.all_lost}")
        print('---'+str(datetime.timedelta(seconds=int(time.time()-self.start_time))
                        )+"---scGNN finished")
        
        return self.all_lost

class Pars:
    def __init__(self):
        self.lr = 1e-3
        self.weight_decay = 1e-3
        self.hidden_dim = 128
        self.hidden_length = 0

    def set_hp(self, hps: dict):
        print(hps)
        for i in hps:
            setattr(self, i, hps[i])

    def check_and_return(self, attribute_name):
        if hasattr(self, attribute_name):
            return getattr(self, attribute_name)
        else:
            return 0
        
    def __str__(self):
        attributes_str = "\n".join(f"{attr}: {value}" for attr, value in self.__dict__.items())
        return f"{self.__class__.__name__} attributes:\n{attributes_str}"


def do_hpo():
    Parser = get_parser()

    CLA = scGNN(Parser)
    PAR = Pars()

    def f(hps):
        PAR.set_hp(hps)

        CLA.get_model(
            hidden_dim = PAR.check_and_return("hidden_dim"),
            hidden_length= PAR.check_and_return("hidden_length")
        )
        CLA.get_optimizer(
            lr=PAR.check_and_return("lr"),
            weight_decay=PAR.check_and_return("weight_decay")
        )
        result = CLA.origin_main()

        print("acc", result)
        return result

    hp_space = [
        LogRangeHP("lr", 1e-4, 1e-1),
        LogRangeHP("weight_decay", 1e-4, 1e-2),
        ChoiceHP("hidden_dim", [64, 128, 192, 256]),
        ChoiceHP("hidden_length", [1, 2, 3, 4, 5])
    ]
    hpo = build_hpo_from_name("tpe", hp_space, f)
    print(hpo.optimize())
    print(str(PAR))

do_hpo()