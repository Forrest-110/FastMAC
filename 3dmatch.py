from data import ThreeDLomatch,ThreeDmatch
from tqdm import tqdm
import torch
import torch.nn as nn
from gconstructor import GraphConstructorFor3DMatch
from gfilter import graphFilter,datasample
import time
import numpy as np
torch.manual_seed(42)


def normalize(x):
    # transform x to [0,1]
    x=x-x.min()
    x=x/x.max()
    return x


def Config():
    config={
        "num_points":np.inf,
        "resolution":0.006,
        "data_dir":'/data/Processed_3dmatch_3dlomatch/',
        "name":"3dmatch",
        'descriptor':'fpfh',
        'batch_size':1,
        'inlier_thresh':0.1,
        'device':'cuda',
        'mode':'graph',
        'ratio':0.50,
    }
    return config

def main():
    config=Config()
    device=config["device"]
    mode=config["mode"]
    sample_ratio=config["ratio"]
    if config["name"]=="3dmatch":
        dataset=ThreeDmatch(num_points=config["num_points"],data_dir=config["data_dir"],descriptor=config["descriptor"])
    elif config["name"]=="3dlomatch":
        dataset=ThreeDLomatch(num_points=config["num_points"],data_dir=config["data_dir"],descriptor=config["descriptor"])
    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0
    )
    if mode == "graph":
        gc=GraphConstructorFor3DMatch()
        print("Start")
        average_time=0
        for i, data_ in enumerate(tqdm(trainloader)):
            time_start=time.time()
            current_points,ground_truth,label,corr_path,gt_path,lb_path=data_
            current_points=current_points.to(device)
            ground_truth=ground_truth.to(device)
            label=label.to(device)
            corr_graph=gc(current_points,config["resolution"],config["name"],config["descriptor"],config["inlier_thresh"])
            degree_signal=torch.sum(corr_graph,dim=-1)

            corr_laplacian=(torch.diag_embed(degree_signal)-corr_graph).squeeze(0)
            corr_scores=graphFilter(degree_signal.transpose(0,1),corr_laplacian,is_sparse=False)
            
            corr_scores=normalize(corr_scores)
            total_scores=corr_scores
            
            k=int(current_points.shape[1]*sample_ratio)
            idxs=datasample(k,False,total_scores)

            time_end=time.time()
            average_time+=time_end-time_start

            samples=current_points.squeeze(0)[idxs,:]
            lb=label.squeeze(0)[idxs].long()
            samples=samples.cpu().numpy()

            
            out_corr_path=corr_path[0].split(".")[0]+"_"+config["mode"]+"_"+str(int(config["ratio"]*100))+".txt"
            our_gt_path=gt_path[0].split(".")[0]+"_"+config["mode"]+"_"+str(int(config["ratio"]*100))+".txt"
            out_lb_path=lb_path[0].split(".")[0]+"_"+config["mode"]+"_"+str(int(config["ratio"]*100))+".txt"
            np.savetxt(out_corr_path,samples)
            np.savetxt(our_gt_path,ground_truth.squeeze(0).cpu().numpy())
            np.savetxt(out_lb_path,lb.cpu().numpy().astype(int),fmt="%d")
        
        print("Average time: ",average_time/len(trainloader)) 
    else:
        raise NotImplementedError
        
        
if __name__ == "__main__":
    main()
