import copy
import pandas as pd
import numpy as np

from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.sparse import diags, csr_matrix

from copy import deepcopy
import warnings

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing

class ConstantTerm():
    
    def __init__(self, n):
        self._x = np.ones(n).reshape(-1,1)     
        
    @property
    def X(self):
        return self._x
    
    def __str__(self):
        return '<%s, %s:%s>' % (self.__class__.__name__)

class LinearTerm():
    
    def __init__(self, df, *idx, mu = None, sig = None, standard = False, log = False):
        self._df = df
        self._idx = list(idx) 
        X = self._df.iloc[:, self._idx].apply(pd.to_numeric).values
        Xmean = None
        Xstd = None
        
        if log == True:
            X = np.log(np.where(X > 0, X, 1.0)) # replace it to 1 then log(1) is 0 
          
        if standard:
            if mu is not None and sig is not None:
                Xmean = mu
                Xstd = sig
            else:
                Xmean = np.mean(X, axis=0)
                Xstd = np.std(X, axis=0)
            
        self._x = X
    
    @property
    def X(self):
        return self._x
    
    def __str__(self):
        return '<%s, %s:%s>' % (self.__class__.__name__)

class DistanceSmoothing:

    def __init__(self, data1, data2=None, ids1=None, ids2=None, attr=None, distance_metric = 'euclidean', 
                 isInverse = False, binary = False, isKernel = False, 
                 initial_scale = None, kernel_function = 'Gaussian', 
                 self_neighboring = False, row_standard = False, 
                 average = False,log = False, standard = False, 
                 lower_bound = None, upper_bound = None, CI_step = None, num_step = 50, mu = None, sig = None):
        
        # Prepare data1 and data2
        self.data1 = data1 # target support
        self.data2 = data2 if data2 is not None else self.data1 # support needs to be changed
        # unique ids for each observation in data1 and data2
        self.ids1 = self.data1[ids1].values if ids1 is not None else np.arange(len(self.data1))
        if data2 is None:
            ids2 = ids1 
        self.ids2 = self.data2[ids2].values if ids2 is not None else np.arange(len(self.data2))

        # Prepare the distance calculation
        self.data1_coords = np.array(list(self.data1.geometry.apply(lambda geom: (geom.x, geom.y))))
        self.data2_coords = np.array(list(self.data2.geometry.apply(lambda geom: (geom.x, geom.y)))) 
        self.tree = cKDTree(self.data2_coords)
        
        # Get the distance matrix 
        self.distance_metric = distance_metric
        distmat = cdist(self.data1_coords, self.data2_coords, metric=self.distance_metric)  
        self._distmat = distmat 
        self._sparse_distmat = csr_matrix(distmat)

        # Initialize weights calulcation 
        self.isInverse = isInverse 
        if self.isInverse:
            self.initial_scale = initial_scale if initial_scale is not None else -1.0
        else:
            self.initial_scale = initial_scale if initial_scale is not None else int(self._distmat.max()/2)
        self.scale = None # to store sigma value used for cal()
        
        self.binary = binary # using 1 or 0
        self.isKernel = isKernel # using kernel function with distance band is the fixed bandwidth
        self._kernel_function = kernel_function.lower()
        
        self.self_neighboring = self_neighboring 
        if data2 is not None and self_neighboring == True:
            # For Suport A to B (i.e., data2 is not empty), there is no concept of self-neighboring.
            warnings.warn(f"There is not concept of self-neighbor for support A to B. Specifying False will.", UserWarning)
            self.self_neighboring = False
        if data2 is None and self_neighboring is None:
            # For Support A to A (i.e., data2 is empty), self-neighboring is required to be specified by users
            warnings.warn(f"Self_neighboring is required to be specified. Defaulting to True will.", UserWarning)
            self.self_neighboring = True
        
        self.row_standard = row_standard
        
        # Prepare variable values calculation
        self.attr = self.data2[attr].values
        self.average = average if not self.binary else False # rowsd is equivelent to average for binary weights 
        self.log = log
        self.standard = standard
        self.mu = mu
        self.sig = sig
        
        # Store weights and neighbors for each observation in data1
        self.neighbors = None
        self.weights = None 

        # For automate selection in GASS
        if isInverse:
            self.lower_bound = lower_bound if lower_bound is not None else -3.0
            self.upper_bound = upper_bound if upper_bound is not None else -0.01
            self.CI_step = CI_step if CI_step is not None else 0.01
        else:
            self.lower_bound = lower_bound if lower_bound is not None else self._distmat.min()
            self.upper_bound = upper_bound if upper_bound is not None else self._distmat.max()
            if num_step is None:
                num_step = int(100)
            self.CI_step = CI_step if CI_step is not None else (self.upper_bound - self.lower_bound) / num_step * 1.0000001
        
    def cal(self, scale):

        dist_w = self._sparse_distmat.copy()
        if scale is None:
            scale = self.initial_scale * 1.00000001
            warnings.warn(f"Scale hyperparemeter is not set. Defaulting to the initial value will.", UserWarning)

        if self.isInverse:
            nonzero_weights = dist_w.power(scale).data 
            dist_w.data = nonzero_weights 
        
        if self.binary:
            dist_w.data = np.where(dist_w.data <= scale, 1, 0) 
            dist_w.eliminate_zeros() # remove extra zeros

        if self.isKernel:            
            fixed_bws = np.full(len(self.data1), scale)
            # filter out neighbors whose distance is larger than the fixed bandwidth
            dist_w.data = np.where(dist_w.data <= scale, dist_w.data, 0) 
            dist_w.eliminate_zeros()
            nonzero_zmat = diags(1/fixed_bws).dot(dist_w)
            nonzero_zmat.data  = _calculate_kernel_weight(self._kernel_function, nonzero_zmat)
            dist_w = nonzero_zmat
                
        if self.row_standard:
            dist_w = _row_standardize_sparse(dist_w)
            
        if self.self_neighboring:
            dist_w.setdiag(1.0)
            
        self.dist_w = dist_w
        self.scale = scale
        
        res = self.dist_w.dot(self.attr)
        
        if self.average:
            row_counts = np.diff(self.dist_w.indptr).flatten()
            row_counts[row_counts == 0] = 1
            res = res / row_counts    
        
        if self.log:
            res = np.log(np.where(res > 0, res, 1.0)) # replace it to 1 then log(1) is 0 
    
        if self.standard:
            Xmean = None
            Xstd = None
            
            if self.mu is not None and self.sig is not None:
                Xmean = self.mu
                Xstd = self.sig
            else:
                Xmean = np.mean(res)
                Xstd = np.std(res)
                
            res = (res - Xmean)/Xstd

        self.res = res
        return res   
    
    def cal2(self, scale_values):
        self.X_cached = {}
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(_cal_worker_thread, self, scale) for scale in scale_values]
            for future in as_completed(futures):
                scale, result = future.result()
                self.X_cached[scale] = result.reshape(-1,1)
                
    def get_cached(self, scale):
        arr = None 
        if self.isInverse:
            arr = self.X_cached[np.round(scale, 3)] 
        else:
            arr = self.X_cached[int(scale)]  
        return arr

class KernelSmoothing:

    def __init__(self, data1, data2=None, ids1=None, ids2=None, attr=None,
                 distance_metric = 'euclidean', binary = False, kernel_function='gaussian',
                 initial_scale = 1, self_neighboring = False, row_standard = False, 
                 average = False, log = False, standard = False, 
                 lower_bound = None, upper_bound = None, CI_step = None, mu = None, sig = None):
        
        # Prepare data1 and data2
        self.data1 = data1 # target support
        self.data2 = data2 if data2 is not None else self.data1 # support needs to be changed
        self.ids1 = self.data1[ids1].values if ids1 is not None else np.arange(len(self.data1))
        if data2 is None:
            ids2 = ids1 
        self.ids2 = self.data2[ids2].values if ids2 is not None else np.arange(len(self.data2))
        
        # Prepare the distance calculation 
        self.data1_coords = np.array(list(self.data1.geometry.apply(lambda geom: (geom.x, geom.y))))
        self.data2_coords = np.array(list(self.data2.geometry.apply(lambda geom: (geom.x, geom.y)))) 
        self.tree = cKDTree(self.data2_coords)

        # Get the distance matrix 
        self.distance_metric = distance_metric
        distmat = cdist(self.data1_coords, self.data2_coords, metric=self.distance_metric)  
        self._distmat = distmat 
        self._sparse_distmat = csr_matrix(distmat)

        # Initialize weights calulcation
        self.binary = binary
        # if not self.binary:
        self.bandwidth = None
        self._kernel_function = kernel_function
            
        self.self_neighboring = self_neighboring
        if data2 is not None and self_neighboring == True:
            # For Suport A to B (i.e., data2 is not empty), there is no concept of self-neighboring.
            warnings.warn(f"There is not concept of self-neighbor for support A to B. Specifying False will.", UserWarning)
            self.self_neighboring = False
        if data2 is None and self_neighboring is None:
            # For Support A to A (i.e., data2 is empty), self-neighboring is required to be specified by users
            warnings.warn(f"Self_neighboring is required to be specified. Defaulting to True will.", UserWarning)
            self.self_neighboring = True
        self.initial_scale = initial_scale
        self.k = None # to store k value used for cal()   
        self.row_standard = row_standard

        # Prepare variable values calculation 
        self.attr = self.data2[attr].values 
        self.average = average if not self.binary else False # rowsd is equivelent to average for binary weights 
        self.log = log
        self.standard = standard
        self.mu = mu
        self.sig = sig
        
        # Store weights and neighbors for each observation in data1
        self.neighbors = None
        self.weights = None 

        # For automate selection in GASS
        self.lower_bound = lower_bound if lower_bound is not None else int(2)
        self.upper_bound = upper_bound if upper_bound is not None else int(len(self.data2))
        self.CI_step = CI_step if CI_step is not None else int(1)
        
    def cal(self, k = None):
        
        if k is None:
            k = self.initial_k
            warnings.warn(f"K is not set. Defaulting to the initial k value will.", UserWarning)

        k_dist_mat, bws = self._get_k_nearest_among_nonzeros(int(k))
        
        if self.binary:
            kernel_w = deepcopy(k_dist_mat)
            kernel_w.data[:] = 1.0
        else:
            self.bandwidth = bws
            nonzero_zmat = (diags(1 / bws).dot(k_dist_mat))
            nonzero_zmat.data = _calculate_kernel_weight(self._kernel_function, nonzero_zmat)
            kernel_w = nonzero_zmat

        if self.row_standard:
            kernel_w = _row_standardize_sparse(kernel_w)
        
        if self.self_neighboring:
            kernel_w.setdiag(1.0)

        self.kernel_w = kernel_w
            
        res = self.kernel_w.dot(self.attr.reshape(-1,1))

        if self.average:
            row_counts = np.diff(self.kernel_w.indptr).reshape(-1, 1) 
            row_counts[row_counts == 0] = 1
            res = res / row_counts
                      
        if self.log:
            res = np.log(np.where(res > 0, res, 1.0)) # replace it to 1 then log(1) is 0 
    
        if self.standard:
            Xmean = None
            Xstd = None
            if self.mu is not None and self.sig is not None:
                Xmean = self.mu
                Xstd = self.sig
            else:
                Xmean = np.mean(res)
                Xstd = np.std(res)

        res = (res-Xmean)/Xstd
        self.res = res
        return res

    def _get_k_nearest_among_nonzeros(self, k):
        
        # initialize
        mat = deepcopy(self._sparse_distmat)
        bws = np.ones(len(self.data1))  
        tmp_k = k if self.self_neighboring else k+1
        
        for i in range(mat.shape[0]):
            row_start = mat.indptr[i]
            row_end = mat.indptr[i + 1]
            row_data = mat.data[row_start:row_end]

            if len(row_data) > k: 
                partition_indices = np.argpartition(row_data, tmp_k - 1) # partion k-th nearest neighbors
                row_data[partition_indices[tmp_k - 1:]] = 0.0 # Mask row data
                mat.data[row_start:row_end] = row_data
            
            bws[i] = np.max(row_data) if len(row_data) > 0 else 1.0

        mat.eliminate_zeros() # guarantee a pure csr matrix
                
        return mat, bws
   
    def cal2(self, scale_values):
        self.X_cached = {}
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(_cal_worker_thread, self, scale) for scale in scale_values]
            for future in as_completed(futures):
                scale, result = future.result()
                self.X_cached[scale] = result.reshape(-1,1)
                
    def get_cached(self, scale):
        return self.X_cached[int(scale)]
    
def _cal_worker_thread(self, scale):
        return scale, self.cal(scale).copy()
    
def _row_standardize_sparse(csr_mat):
    row_sums = np.array(csr_mat.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1 #prevents division by zero
    return csr_mat.multiply(1 / row_sums[:, np.newaxis]).tocsr()

def _calculate_kernel_weight(kernel_function, input_data):
               
    z = input_data.data if isinstance(input_data, csr_matrix) else input_data

    if kernel_function == 'bisquare':
        weights = (1 - z ** 2)**2
        
    elif kernel_function == 'gaussian':
        c = (np.pi * 2) ** (-0.5)
        weights = c * np.exp(-(z ** 2) / 2.0)
        
    elif kernel_function == 'quadratic':
        weights = (3.0/4) * (1 - z ** 2)
        
    elif kernel_function == 'quartic':
        weights = (15.0/16) * (1 - z ** 2)**2

    elif kernel_function == 'triangular':
        weights = 1 - z #[1 - zi for zi in z]

    elif kernel_function == 'uniform':
        weights = np.ones_like(z) * 0.5

    elif kernel_function == 'knn':
        weights = np.ones_like(z) * 1.0

    else:
        raise ValueError(f"Unsupported kernel function: {_kernel_function}")
    
    return weights