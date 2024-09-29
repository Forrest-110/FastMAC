from .fastmac.gconstructor import GraphConstructor
from .fastmac.gfilter import graphFilter,datasample
import torch
def Config():
    config={
        "num_points":5000,
        "data_dir":'/data/Processed_KITTI/correspondence_fcgf/',
        "filename":'fcgf@corr.txt',
        'gtname':'fcgf@gtmat.txt',
        'labelname':'fcgf@gtlabel.txt',
        'batch_size':1,
        'inlier_thresh':0.6,
        'thresh':0.999,
        'sigma':0.6,
        'tau':0.,
        'device':'cuda',
        'mode':'graph',
        'ratio':0.01,
        'outpath':'/home/Zero/mac/ablation/score_weight/111/1/sample',
        'pc1_weight':1,
        'pc2_weight':1,
        'degree_weight':1,
    }
    return config

config = Config()
gc=GraphConstructor(config["inlier_thresh"],config["thresh"],trainable=False,sigma=config["sigma"],tau=config["tau"])

def normalize(x):
    # transform x to [0,1]
    x=x-x.min()
    x=x/x.max()
    return x

def downsample(corr):
    if len(corr.shape)==2:
        corr=corr.unsqueeze(0)
    corr_graph=gc(corr,mode="correspondence")
    pc1_signal=corr[:,:,:3]
    pc2_signal=corr[:,:,3:]
    degree_signal=torch.sum(corr_graph,dim=-1)
    pc_graph1=gc(pc1_signal,mode="pointcloud")
    pc_graph2=gc(pc2_signal,mode="pointcloud")

    pc_laplacian1=(torch.diag_embed(torch.sum(pc_graph1,dim=-1))-pc_graph1).squeeze(0)
    pc_laplacian2=(torch.diag_embed(torch.sum(pc_graph2,dim=-1))-pc_graph2).squeeze(0)

    pc1_scores=graphFilter(pc1_signal.squeeze(0),pc_laplacian1,is_sparse=False)
    pc2_scores=graphFilter(pc2_signal.squeeze(0),pc_laplacian2,is_sparse=False)

    corr_laplacian=(torch.diag_embed(degree_signal)-corr_graph).squeeze(0)
    
    
    corr_scores=graphFilter(degree_signal.transpose(0,1),corr_laplacian,is_sparse=False)
    # corr_scores=graphFilter(torch.ones_like(degree_signal.transpose(0,1)).cuda(),torch.matmul(corr_laplacian,corr_laplacian),is_sparse=False)
    
    pc1_scores=normalize(pc1_scores)
    pc2_scores=normalize(pc2_scores)
    corr_scores=normalize(corr_scores)

    total_scores=config["pc1_weight"]*pc1_scores+config["pc2_weight"]*pc2_scores+config["degree_weight"]*corr_scores
    idxs=datasample(1000,False,total_scores)
    samples=corr.squeeze(0)[idxs,:]
    return samples