from data import processedKITTI
from tqdm import tqdm
import torch
from gconstructor import GraphConstructor
from gfilter import graphFilter,datasample
import time
import numpy as np
import os
torch.manual_seed(42)


def normalize(x):
    # transform x to [0,1]
    x=x-x.min()
    x=x/x.max()
    return x

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
        'outpath':'',
    }
    return config

def main():
    config=Config()
    device=config["device"]
    mode=config["mode"]
    sample_ratio=config["ratio"]
    dataset=processedKITTI(config["num_points"],config["data_dir"],config["filename"],config["gtname"],config["labelname"])
    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0
    )
    if mode == "graph":
        gc=GraphConstructor(config["inlier_thresh"],config["thresh"],trainable=False,sigma=config["sigma"],tau=config["tau"])
        print("Start")
        average_time=0
        for i, data_ in enumerate(tqdm(trainloader)):
            time_start=time.time()
            pts,gt,lb=data_
            pts=pts.to(device)
            gt=gt.to(device)
            lb=lb.to(device)
            corr_graph=gc(pts,mode="correspondence")
            degree_signal=torch.sum(corr_graph,dim=-1)

            
            
            corr_laplacian=(torch.diag_embed(degree_signal)-corr_graph).squeeze(0)
            corr_scores=graphFilter(degree_signal.transpose(0,1),corr_laplacian,is_sparse=False) 
            
            
            total_scores=corr_scores
            k=int(config["num_points"]*sample_ratio)
            idxs=datasample(k,False,total_scores)

            time_end=time.time()
            
            average_time+=time_end-time_start

            samples=pts.squeeze(0)[idxs,:]
            lb=lb.squeeze(0)[idxs].long()
            samples=samples.cpu().numpy()

            outdir=os.path.join(config["outpath"],str(i))
            if not os.path.exists((outdir)):
                os.makedirs((outdir))

            np.savetxt(outdir+'/'+config["filename"],samples)
            np.savetxt(outdir+'/'+config["gtname"],gt.squeeze(0).cpu().numpy())
            np.savetxt(outdir+'/'+config["labelname"],lb.cpu().numpy().astype(int),fmt="%d")
        
        print("Average time: ",average_time/len(trainloader))
    else:
        raise NotImplementedError
        
        
if __name__ == "__main__":
    main()
