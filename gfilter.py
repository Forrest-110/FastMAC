import torch

torch.manual_seed(42)

def graphFilter(points,adjacent_matrix,is_sparse):
    '''
    points: n x 3
    adjacent_matrix: sparse matrix
    
    return:
    score: n x 1
    '''
    if is_sparse:
        xyz=torch.sparse.mm(adjacent_matrix,points)
    else:
        xyz=torch.mm(adjacent_matrix,points)
    return torch.norm(xyz,dim=-1)

def graphLowFilter(points,adjacent_matrix):
    '''
    points: n x 3
    adjacent_matrix: sparse matrix
    
    return:
    score: n x 1
    '''
    r=torch.matmul(torch.eye(points.shape[0]).to(adjacent_matrix.device)+adjacent_matrix, points)
    return torch.norm(r,p=2,dim=-1)

def graphAllPassFilter(points):
    '''
    points: n x 3
    adjacent_matrix: sparse matrix
    
    return:
    score: n x 1
    '''
    return torch.norm(points,p=2,dim=-1)


def datasample(k,replace,weights):
    '''
    idxs: n
    k: int
    replace: bool
    weights: n
    '''
    return torch.multinomial(weights,k,replacement=replace)


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
    
    pc=read_ply('/data/cubic.ply')
    num_pts=pc.shape[0]
    sample_rate=0.25
    k=np.floor(sample_rate*num_pts)
    print(k)

