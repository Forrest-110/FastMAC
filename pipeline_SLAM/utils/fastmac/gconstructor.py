import torch
import torch.nn as nn

def euclidean(a, b):
    return torch.norm(a - b, dim=-1, keepdim=True)

def compatibility(a,b):
    assert(a.shape[-1]==6)
    assert(b.shape[-1]==6)
    n1=torch.norm(a[...,:3]-b[...,:3],dim=-1,keepdim=True)
    n2=torch.norm(a[...,3:]-b[...,3:],dim=-1,keepdim=True)
    return torch.abs(n1-n2)

def Dmatrix(a,type):
    if type=="euclidean":
        return torch.cdist(a,a)
        
    elif type=="compatibility":
        a1=a[...,:3]
        a2=a[...,3:]
        return torch.abs(Dmatrix(a1,"euclidean")-Dmatrix(a2,"euclidean"))

class GraphConstructor(nn.Module):
    def __init__(self,inlier_thresh,thresh,trainable,device="cuda",sigma=None,tau=None) -> None:
        '''
        inlier thresh: KITTI 0.6, 3dmatch 0.1
        thresh: fpfh 0.9, fcgf 0.999
        '''
        super().__init__()
        self.device=device
        self.inlier_thresh=nn.Parameter(torch.tensor(inlier_thresh,requires_grad=trainable,dtype=torch.float32)).to(device)
        self.thresh=nn.Parameter(torch.tensor(thresh,requires_grad=trainable,dtype=torch.float32)).to(device)
        if sigma is not None:
            self.sigma=nn.Parameter(torch.tensor(sigma,requires_grad=trainable,dtype=torch.float32)).to(device)
        else:
            self.sigma=self.inlier_thresh
        if tau is not None:
            self.tau=nn.Parameter(torch.tensor(tau,requires_grad=trainable,dtype=torch.float32)).to(device)
        else:
            self.tau=self.thresh
    def forward(self,points,mode,k1=2,k2=1):
        '''
        points: B x M x 6
        output: B x M x M
        '''
        if mode=="correspondence":
            points=points.to(self.device)
            dmatrix=Dmatrix(points,"compatibility")
            score=1-dmatrix**2/self.inlier_thresh**2
            # score=torch.exp(-dmatrix**2/self.inlier_thresh**2)
            score[score<self.thresh]=0
            if k1==1:
                return score
            else:
                return score*torch.einsum("bmn,bnk->bmk",score,score)
        elif mode=="pointcloud":
            '''
            points: B x N x 3
            output: B x N x N
            '''
            points=points.to(self.device)
            dmatrix=Dmatrix(points,"euclidean")
            
            # score=1-dmatrix**2/self.inlier_thresh**2
            score=torch.exp(-dmatrix**2/self.sigma**2)
            score[score<self.tau]=0
            if k2==1:
                return score
            else:
                return score*torch.einsum("bmn,bnk->bmk",score,score)
        
class GraphConstructorFor3DMatch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        pass
    def forward(self,correspondence, resolution,  name, descriptor, inlier_thresh):
        self.device="cuda"
        correspondence=correspondence.to(self.device)
        dmatrix=Dmatrix(correspondence,"compatibility")

        if descriptor=="predator":
            score=1-dmatrix**2/inlier_thresh**2
            score[score<0.999]=0
        else:
            alpha_dis = 10 * resolution
            score = torch.exp(-dmatrix**2 / (2 * alpha_dis * alpha_dis))
            if (name == "3dmatch" and descriptor == "fcgf"):
                score[score<0.999]=0
            elif (name == "3dmatch" and descriptor == "fpfh") :
                score[score<0.995]=0
            elif (descriptor == "spinnet" or descriptor == "d3feat") :
                score[score<0.85]=0
                        #spinnet 5000 2500 1000 500 250
                        #         0.99 0.99 0.95 0.9 0.85
            else:
                score[score<0.99]=0 #3dlomatch 0.99, 3dmatch fcgf 0.999 fpfh 0.995
        return score*torch.einsum("bmn,bnk->bmk",score,score)


class Graph:
    def __init__(self):
        pass

    @staticmethod
    def construct_graph(pcloud, nb_neighbors):
        """
        Construct a directed nearest neighbor graph on the input point cloud.

        Parameters
        ----------
        pcloud : torch.Tensor
            Input point cloud. Size B x N x 3.
        nb_neighbors : int
            Number of nearest neighbors per point.

        Returns
        -------
        graph : flot.models.graph.Graph
            Graph build on input point cloud containing the list of nearest 
            neighbors (NN) for each point and all edge features (relative 
            coordinates with NN).
            
        """

        # Size
        nb_points = pcloud.shape[1]
        size_batch = pcloud.shape[0]

        # Distance between points
        distance_matrix = torch.sum(pcloud ** 2, -1, keepdim=True)
        distance_matrix = distance_matrix + distance_matrix.transpose(1, 2)
        distance_matrix = distance_matrix - 2 * torch.bmm(
            pcloud, pcloud.transpose(1, 2)
        )
        # except self distance
        distance_matrix = distance_matrix + 1e6 * torch.eye(nb_points).unsqueeze(0).to(pcloud.device)

        # Find nearest neighbors
        neighbors = torch.argsort(distance_matrix, -1)[..., :nb_neighbors]

        # # direclty construct dense adjacency matrix
        # adj_dense=torch.zeros((nb_points,nb_points)).to(pcloud.device)
        # for i in range(nb_points):
        #     for j in range(nb_neighbors):
        #         adj_dense[i,neighbors[0,i,j]]=1
        

        # construct sparse adjacency matrix
        neighbors_flat = neighbors.reshape( -1)
        idx=torch.arange(nb_points).repeat(nb_neighbors,1).transpose(0,1).reshape(-1)
        idx=idx.to(pcloud.device)
        neighbors_flat=neighbors_flat.to(pcloud.device)
        i=torch.stack([idx,neighbors_flat],dim=0)
        v=torch.ones(i.shape[1]).to(pcloud.device)
        print(i)
        adj=torch.sparse_coo_tensor(i,v,(nb_points,nb_points))

        # assert(torch.all(torch.eq(adj.to_dense(),adj_dense)))

        return adj
        
        
    

if __name__ == "__main__":
    from plyfile import PlyData,PlyElement
    import numpy as np
    def write_ply(save_path,points,text=True):
        """
        save_path : path to save: '/yy/XX.ply'
        pt: point_cloud: size (N,3)
        """
        points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
        vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
        el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
        PlyData([el], text=text).write(save_path)
    def read_ply(filename):
        """ read XYZ point cloud from filename PLY file """
        plydata = PlyData.read(filename)
        pc = plydata['vertex'].data
        pc_array = np.array([[x, y, z] for x,y,z in pc])
        return pc_array
    
    pc=read_ply('/data/plane2.ply')
    num_pts=pc.shape[0]
    sample_rate=0.01
    k=int(np.floor(sample_rate*num_pts))
    pc_tensor=torch.from_numpy(pc).type(torch.FloatTensor).unsqueeze(0).cuda()
    g=GraphConstructor(0.6,0,False)
    adj=g(pc_tensor,"pointcloud")
    
    degree=torch.diag_embed(torch.sum(adj,dim=-1))
    laplacian=(degree-adj).squeeze(0)
    low_shift=(torch.diag_embed(1/torch.sum(adj,dim=-1))*adj).squeeze(0)
    from gfilter import graphLowFilter,datasample
    scores=graphLowFilter(pc_tensor.squeeze(0),low_shift)
    idxs=datasample(k,False,scores)
    sampled_pc=pc_tensor.squeeze(0)[idxs,:]
    write_ply('/data/plane2_sampled_low.ply',sampled_pc.cpu().numpy())
