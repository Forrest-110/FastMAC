import torch
import torch.utils.data as data
import numpy as np
import os
import sys
import open3d as o3d
torch.manual_seed(42)
sys.path.append(os.path.dirname(__file__))

class processedKITTI(data.Dataset):
    def __init__(self,num_points,data_dir,filename,gtname,labelname,num_samples=-1) -> None:
        super().__init__()
        self.data_dir=data_dir
        self.num_points=num_points
        self.filedirs=os.listdir(data_dir)
        self.filenames=[os.path.join(data_dir,filedir,filename) for filedir in self.filedirs]
        self.gtnames=[os.path.join(data_dir,filedir,gtname) for filedir in self.filedirs]
        self.labelnames=[os.path.join(data_dir,filedir,labelname) for filedir in self.filedirs]
        self.filenames.sort()
        self.gtnames.sort()
        self.labelnames.sort()
        if num_samples>0:
            self.filenames=self.filenames[:num_samples]
            self.gtnames=self.gtnames[:num_samples]
            self.labelnames=self.labelnames[:num_samples]
       

    def __getitem__(self, index):
        filename=self.filenames[index]
        gtname=self.gtnames[index]
        labelname=self.labelnames[index]
        data=np.loadtxt(filename,delimiter=' ')
        ground_truth=np.loadtxt(gtname,delimiter=' ')
        label=np.loadtxt(labelname,delimiter=' ')
        n_pts=data.shape[0]
        num_points=min(n_pts,self.num_points)
        pt_idxs=np.arange(0,n_pts)
        np.random.shuffle(pt_idxs)
        current_points=data[pt_idxs[:num_points],:].copy()
        current_points=torch.from_numpy(current_points).type(torch.FloatTensor)
        ground_truth=torch.from_numpy(ground_truth).type(torch.FloatTensor)
        label=torch.from_numpy(label).type(torch.FloatTensor)
        return current_points,ground_truth,label
    def __len__(self):
        return len(self.filenames)

class ThreeDmatch(data.Dataset):
    def __init__(self,num_points,data_dir,descriptor,num_samples=-1) -> None:
        super().__init__()
        self.data_dir=data_dir
        self.num_points=num_points
    
        self.data_scenes=[
	
	"7-scenes-redkitchen",
	"sun3d-home_at-home_at_scan1_2013_jan_1",
	"sun3d-home_md-home_md_scan9_2012_sep_30",
	"sun3d-hotel_uc-scan3",
	"sun3d-hotel_umd-maryland_hotel1",
	"sun3d-hotel_umd-maryland_hotel3",
	"sun3d-mit_76_studyroom-76-1studyroom2",
	"sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika",

        ]
        self.descriptor=descriptor
        self.filenames=[]
        self.gtnames=[]
        self.labelnames=[]
        self.srcply=[]
        self.tgtply=[]
        for data_scene in self.data_scenes:
            if (descriptor == "fpfh" or descriptor == "spinnet" or descriptor == "d3feat"):
                loadertxt=data_scene+"/dataload.txt"
            elif descriptor == "fcgf":
                loadertxt=data_scene+"/dataload_fcgf.txt"
            loadertxt=data_dir+'/'+loadertxt
            with open(loadertxt,'r') as f:
                lines=f.readlines()
                for line in lines:
                    line=line.strip('\n')
                    src_ply=data_dir+'/'+data_scene+line.split('+')[0]+".ply"
                    tgt_ply=data_dir+'/'+data_scene+line.split('+')[1]+".ply"
                    corr_path=data_dir+'/'+data_scene+'/'+line+("@corr_fcgf.txt" if descriptor == "fcgf" else "@corr.txt")
                    gt_path=data_dir+'/'+data_scene+'/'+line+("@GTmat_fcgf.txt" if descriptor == "fcgf" else "@GTmat.txt")
                    label_path=data_dir+'/'+data_scene+'/'+line+("@label_fcgf.txt" if descriptor == "fcgf" else "@label.txt")
                    self.srcply.append(src_ply)
                    self.tgtply.append(tgt_ply)
                    self.filenames.append(corr_path)
                    self.gtnames.append(gt_path)
                    self.labelnames.append(label_path)

        if num_samples>0:
            self.srcply=self.srcply[:num_samples]
            self.tgtply=self.tgtply[:num_samples]
            self.filenames=self.filenames[:num_samples]
            self.gtnames=self.gtnames[:num_samples]
            self.labelnames=self.labelnames[:num_samples]
       

    def __getitem__(self, index):
        src_ply=self.srcply[index]
        tgt_ply=self.tgtply[index]
        filename=self.filenames[index]
        gtname=self.gtnames[index]
        labelname=self.labelnames[index]
        data=np.loadtxt(filename,delimiter=' ')
        ground_truth=np.loadtxt(gtname,delimiter=' ')
        label=np.loadtxt(labelname,delimiter=' ')
        n_pts=data.shape[0]
        num_points=min(n_pts,self.num_points)
        pt_idxs=np.arange(0,n_pts)
        np.random.shuffle(pt_idxs)
        current_points=data[pt_idxs[:num_points],:].copy()
        current_points=torch.from_numpy(current_points).type(torch.FloatTensor)
        ground_truth=torch.from_numpy(ground_truth).type(torch.FloatTensor)
        label=torch.from_numpy(label).type(torch.FloatTensor)
        return current_points,ground_truth,label,filename,gtname,labelname
    def __len__(self):
        return len(self.filenames)   

class ThreeDLomatch(data.Dataset):
    def __init__(self,num_points,data_dir,descriptor,num_samples=-1) -> None:
        super().__init__()
        self.data_dir=data_dir
        self.num_points=num_points
    
        self.data_scenes=[
	"7-scenes-redkitchen_3dlomatch",
	"sun3d-home_at-home_at_scan1_2013_jan_1_3dlomatch",
	"sun3d-home_md-home_md_scan9_2012_sep_30_3dlomatch",
	"sun3d-hotel_uc-scan3_3dlomatch",
	"sun3d-hotel_umd-maryland_hotel1_3dlomatch",
	"sun3d-hotel_umd-maryland_hotel3_3dlomatch",
	"sun3d-mit_76_studyroom-76-1studyroom2_3dlomatch",
	"sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika_3dlomatch",
        ]
        self.descriptor=descriptor
        self.filenames=[]
        self.gtnames=[]
        self.labelnames=[]
        self.srcply=[]
        self.tgtply=[]
        for data_scene in self.data_scenes:
            if (descriptor == "fpfh" or descriptor == "spinnet" or descriptor == "d3feat"):
                loadertxt=data_scene+"/dataload.txt"
            elif descriptor == "fcgf":
                loadertxt=data_scene+"/dataload_fcgf.txt"
            loadertxt=data_dir+'/'+loadertxt
            with open(loadertxt,'r') as f:
                lines=f.readlines()
                for line in lines:
                    line=line.strip('\n')
                    src_ply=data_dir+'/'+data_scene+line.split('+')[0]+".ply"
                    tgt_ply=data_dir+'/'+data_scene+line.split('+')[1]+".ply"
                    corr_path=data_dir+'/'+data_scene+'/'+line+("@corr_fcgf.txt" if descriptor == "fcgf" else "@corr.txt")
                    gt_path=data_dir+'/'+data_scene+'/'+line+("@GTmat_fcgf.txt" if descriptor == "fcgf" else "@GTmat.txt")
                    label_path=data_dir+'/'+data_scene+'/'+line+("@label_fcgf.txt" if descriptor == "fcgf" else "@label.txt")
                    self.srcply.append(src_ply)
                    self.tgtply.append(tgt_ply)
                    self.filenames.append(corr_path)
                    self.gtnames.append(gt_path)
                    self.labelnames.append(label_path)

        if num_samples>0:
            self.srcply=self.srcply[:num_samples]
            self.tgtply=self.tgtply[:num_samples]
            self.filenames=self.filenames[:num_samples]
            self.gtnames=self.gtnames[:num_samples]
            self.labelnames=self.labelnames[:num_samples]
       

    def __getitem__(self, index):
        src_ply=self.srcply[index]
        tgt_ply=self.tgtply[index]
        filename=self.filenames[index]
        gtname=self.gtnames[index]
        labelname=self.labelnames[index]
        data=np.loadtxt(filename,delimiter=' ')
        ground_truth=np.loadtxt(gtname,delimiter=' ')
        label=np.loadtxt(labelname,delimiter=' ')
        n_pts=data.shape[0]
        num_points=min(n_pts,self.num_points)
        pt_idxs=np.arange(0,n_pts)
        np.random.shuffle(pt_idxs)
        current_points=data[pt_idxs[:num_points],:].copy()
        current_points=torch.from_numpy(current_points).type(torch.FloatTensor)
        ground_truth=torch.from_numpy(ground_truth).type(torch.FloatTensor)
        label=torch.from_numpy(label).type(torch.FloatTensor)
        return current_points,ground_truth,label,filename,gtname,labelname
    def __len__(self):
        return len(self.filenames)

    

if __name__ == "__main__":
    from tqdm import tqdm
    # num_points=5000
    # data_dir='/data/Processed_KITTI/correspondence_fpfh/'
    # filename='fpfh@corr.txt'
    # gtname='fpfh@gtmat.txt'
    # labelname='fpfh@gtlabel.txt'
    # dataset=processedKITTI(num_points,data_dir,filename,gtname,labelname)
    # trainloader = torch.utils.data.DataLoader(
    #     dataset, batch_size=1, shuffle=False, num_workers=0
    # )
    # for i, data_ in enumerate(tqdm(trainloader)):
    #     pts,gt,label=data_
    #     print(label.shape)
    #     break

    data_dir='/data/Processed_3dmatch_3dlomatch/'
    descriptor='fpfh'
    num_points=np.inf
    dataset=ThreeDmatch(num_points,data_dir,descriptor)
    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )
    for i, data_ in enumerate(tqdm(trainloader)):
        pts,gt,label,corr_path,gt_path,lb_path=data_

        print(label.shape)
        print(gt.shape)
        print(pts.shape)

        break