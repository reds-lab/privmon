from os import close
from time import sleep
import numpy as np
import logging
import random
import torch
import lpips
import faiss

from monitor.utils import get_logger
from .config import *

logger = get_logger(__name__, level=logging.DEBUG)

# Take a vector x and returns the indices of its K nearest neighbors in the training set: train_data
class CalDist:
    def __init__(self, net="alex", device="cuda:0"):
        # closer to "traditional" perceptual loss, when used for optimization
        self.loss_fn = lpips.LPIPS(net=net).to(device)
        self.device = device
        self.orig_queries = []
        self.feat_vec_flats = []
        self.initial_flag = True
        self.lsh_idx_reset = False
        self.lsh_counter = 0
        self.lsh_reset = False
        # Preserve high-likely malicious queries
        self.mal_queries = []
        self.mal_dists = []
        # HNSW parameters
        # The number of neighbors
        self.M = 32
        # depth of layers explored during search
        self.ef_search = 32
        # depth of layers explored during index construction
        self.ef_construction = 16
        # store the history query for perc KNN
        self.queries_cat = None

    def hamming_dist(self, a, b):
        a = np.where(a>0.5, 1, 0)
        b = np.where(b>0.5, 1, 0)
        ham = np.nonzero(a-b)
        return np.shape(ham[0])[0]


    # Take a vector x and returns the indices of its K nearest neighbors in the training set: train_data
    def cal_norm(self, query, queries, k=K, shift=1, ord=2):
        queries_cat = torch.cat(queries).to(self.device)
        query = query.to(self.device)
        # use torch.linalg.vector_norm instead of np.linalg.norm
        l2_dist = torch.linalg.vector_norm(queries_cat-query, ord=ord, dim=(-3, -2, -1))
        # Find the K+1 NNs, then exclude itself
        neighbors = torch.sort(l2_dist)[0][shift:k+shift]
        # print(neighbors)
        avg_l2_dist = torch.mean(neighbors)
        return avg_l2_dist.cpu().numpy().item()


    # cal the hamming distance
    def cal_ham(self, query, queries, k=K, shift=1):
        queries_cat = torch.cat(queries).to(self.device)
        # normalize to [0, 1]
        query = torch.where(query>0.5, 1, 0).to(self.device)
        queries_cat = torch.where(queries_cat>0.5, 1, 0)
        # ham_dist = [self.hamming_dist(query, q) for q in queries]
        ham_dist = torch.linalg.vector_norm(queries_cat-query, ord=0, dim=(-3, -2, -1))
        neighbors = torch.sort(torch.tensor(ham_dist, dtype=torch.float))[0][shift:k+shift]
        avg_ham_dist = torch.mean(neighbors)
        return avg_ham_dist.cpu().numpy().item()


    # Cal the average perceptual distance
    def cal_perc(self, query, queries, k=K, shift=1):
        queries_cat = torch.cat(queries, dim=0)
        percep_loss = self.loss_fn(query, queries_cat)
        percep_loss_all = torch.squeeze(percep_loss)
        neighbors = torch.sort(percep_loss_all)[0][shift:k+shift]
        # print("Percuptual distance is called.")
        avg_percept_dist = torch.mean(neighbors)
        return avg_percept_dist.detach().cpu().numpy().item()

    # KNN based on perceptual similarity without sliding window
    # Stateful Detection et. (SPAI'20)
    def cal_perc_knn(self, query, shift=0):
        k = 50
        res = 100
        if self.queries_cat == None:
            self.queries_cat = torch.cat([query], dim=0)
            return res
        if self.queries_cat.shape[0] > k:
            percep_loss = self.loss_fn(query, self.queries_cat)
            percep_loss_all = torch.squeeze(percep_loss)
            neighbors = torch.sort(percep_loss_all)[0][shift:k+shift]
            # print("Percuptual distance is called.")
            avg_percept_dist = torch.mean(neighbors)
            res = avg_percept_dist.detach().cpu().numpy().item()
        self.queries_cat = torch.cat([self.queries_cat, query], dim=0)
        return res

    # KNN based on perceptual similarity without sliding window
    # Stateful Detection et. (SPAI'20)
    def cal_perc_knn_v2(self, query, shift=0):
        k = 50
        res = 100
        # feat_vec is a list of feature vector, the dimension depends on the model.
        feat_vec = self.loss_fn.feat(query.to(self.device))
        # Convert the list to one vector.
        feat_vec_flat = torch.cat(feat_vec, dim=1).cpu()
        if self.queries_cat == None:
            self.queries_cat = torch.cat([feat_vec_flat], dim=0)
            return res
        if self.queries_cat.shape[0] > k:
            percep_loss = torch.linalg.vector_norm(self.queries_cat-feat_vec_flat, ord=2, dim=(1,))
            neighbors = torch.sort(percep_loss)[0][shift:k+shift]
            # print("Percuptual distance is called.")
            avg_percept_dist = torch.mean(neighbors)
            res = avg_percept_dist.detach().cpu().numpy().item()
        self.queries_cat = torch.cat([self.queries_cat, feat_vec_flat], dim=0)
        return res

    def __compute_distance(self, idx, dist, query, k, use_orig, metric="perc", optimized=False, level2=False):
        if use_orig:
            query_list = [self.orig_queries[i] for i in idx[0]]
            queries_cat = torch.cat(query_list, dim=0)
            if metric == "perc":
                loss_values = self.loss_fn(query, queries_cat)
                loss_values = torch.squeeze(loss_values)
            elif metric == "l2":
                loss_values = torch.linalg.vector_norm(queries_cat-query, ord=2, dim=(-3, -2, -1))
            else:
                return dist[0].mean()
            if level2:
                values, _ = loss_values.topk(k, largest=False)
                score = values.mean().detach().cpu().numpy()
            else:
                score = torch.mean(loss_values).detach().cpu().numpy()
            self.orig_queries.append(query)
            # Find the nearest one, then the both of them into mal_queries
            min_dist = loss_values.min().item()
            if optimized and min_dist < LSH_MAL_THOLD:
                closest_idx = idx[0][loss_values.argmin().tolist()]
                closest_query = self.orig_queries[closest_idx]
                self.mal_queries.append(closest_query)
                # self.mal_queries.append(query)
        else:
            score = dist[0].mean()
        return score


    def __init_index(self, d, k, shape, use_orig, hnsw):
        if self.initial_flag:
            self.initial_flag = False
            # Add hnsw support
            if hnsw:
                self.index = faiss.IndexHNSWFlat(d, self.M)
                self.index.hnsw.efConstruction = self.ef_construction
                self.index.hnsw.efSearch = self.ef_search
                self.overlap_index = faiss.IndexHNSWFlat(d, self.M)
                self.overlap_index.hnsw.efConstruction = self.ef_construction
                self.overlap_index.hnsw.efSearch = self.ef_search
            else:
                self.index = faiss.IndexLSH(d, 2048)
                self.overlap_index = faiss.IndexLSH(d, 2048)
            self.lsh_counter = k
            self.index.add(np.zeros((k, d)).astype(np.float32))
            if use_orig:
                self.orig_queries = []
                for _ in range(k):
                    self.orig_queries.append(torch.zeros(shape).to(self.device))
            # print(len(self.orig_queries))
            # print(self.index.ntotal)
        else:
            self.index = self.overlap_index
            if hnsw:
                self.overlap_index = faiss.IndexHNSWFlat(d, self.M)
                self.overlap_index.hnsw.efConstruction = self.ef_construction
                self.overlap_index.hnsw.efSearch = self.ef_search
            else:
                self.overlap_index = faiss.IndexLSH(d, 2048)
            self.lsh_counter = LSH_OVERLAP_SIZE
            if use_orig: self.orig_queries = self.orig_queries[-LSH_OVERLAP_SIZE:]
            # print(len(self.orig_queries))
            # print(self.index.ntotal)
            self.lsh_idx_reset = False


    def __reset_index(self, optimized, k):
        logger.debug(f"Malicious queries number: {len(self.mal_queries)}")
        if (not optimized) or len(self.mal_queries) == 0:
            self.lsh_counter = LSH_OVERLAP_SIZE
            self.orig_queries = self.orig_queries[-LSH_OVERLAP_SIZE:]
            if not optimized:
                self.feat_vec_flats = self.feat_vec_flats[-LSH_OVERLAP_SIZE:]
                # It is inefficient if the overlap size is large
                for fvf in self.feat_vec_flats:
                    self.index.add(fvf)
            else:
                raise ValueError("The malicious queries number is 0")
        else:
            self.lsh_counter = len(self.mal_queries)
            self.orig_queries = self.mal_queries
            self.mal_queries = []
            if self.lsh_counter <= k:
                for _ in range(self.lsh_counter, k):
                    self.orig_queries.append(torch.zeros(self.orig_queries[0].shape).to(self.device))
                self.lsh_counter = k
            for q in self.orig_queries:
                feat_vec = self.loss_fn.feat(q)
                feat_vec_flat = torch.cat(feat_vec, dim=1).cpu().numpy()
                self.index.add(feat_vec_flat)


    # Cal perceptual distance based on lsh
    def cal_perc_lsh(self, query, k=K, use_orig=False, hnsw=False, level2=False, selective=False):
        # set k value based on level2
        real_k = k
        if level2:
            real_k = k * 5
        # feat_vec is a list of feature vector, the dimension depends on the model.
        feat_vec = self.loss_fn.feat(query.to(self.device))
        # convert the list to one vector.
        feat_vec_flat = torch.cat(feat_vec, dim=1).cpu().numpy()
        # Initiate or reset the buffer
        if self.initial_flag or self.lsh_counter == LSH_WINDOW_SIZE:
            d = feat_vec_flat.shape[1]
            self.__init_index(d, real_k, query.shape, use_orig, hnsw)
        # Search in the index
        dist, idx = self.index.search(feat_vec_flat, k=real_k)
        # Compute the distance
        score = self.__compute_distance(idx, dist, query, k, use_orig, optimized=selective, level2=level2)
        # Update the index
        self.index.add(feat_vec_flat)
        if self.lsh_counter == LSH_WINDOW_SIZE - LSH_OVERLAP_SIZE:
            self.lsh_idx_reset = True
        if self.lsh_idx_reset:
            self.overlap_index.add(feat_vec_flat)
        self.lsh_counter += 1
        return score


    # Cal perceptual distance based on lsh version 2
    def cal_perc_lsh_slide(self, query, k=K, use_orig=False,
            selective=False, hnsw=False, level2=False):
        '''
         The only difference is the lsh function initiating method.
         Bascially, this version supports arbitrary overlapping size.
         But the initiating process will spend longer time every
         (window size - overlapping size) queries.
         Besides, this approach has to maintain a feat_vec list.
         optimized: if use the optimized method
         label: if use, input the label
        '''
        # Set k value based on level2
        real_k = k
        if level2:
            real_k = k * 5
        # feat_vec is a list of feature vector, the dimension depends on the model.
        feat_vec = self.loss_fn.feat(query.to(self.device))
        # Convert the list to one vector.
        feat_vec_flat = torch.cat(feat_vec, dim=1).cpu().numpy()
        # Initiate or reset the buffer
        if self.initial_flag or self.lsh_counter == LSH_WINDOW_SIZE:
            d = feat_vec_flat.shape[1]
            # Choose the index
            if hnsw:
                self.index = faiss.IndexHNSWFlat(d, self.M)
                self.index.hnsw.efConstruction = self.ef_construction
                self.index.hnsw.efSearch = self.ef_search
            else:
                self.index = faiss.IndexLSH(d, 2048)
            # Whether initial or not
            if self.initial_flag:
                self.initial_flag = False
                self.lsh_counter = real_k
                self.index.add(np.zeros((real_k, d)).astype(np.float32))
                if use_orig:
                    self.orig_queries = []
                self.feat_vec_flats = []
                for _ in range(real_k):
                    if use_orig:
                        self.orig_queries.append(torch.zeros(query.shape).to(self.device))
                    self.feat_vec_flats.append(np.zeros((1, d)).astype(np.float32))
            else:
                self.__reset_index(selective, real_k)
        # Search in the index
        dist, idx = self.index.search(feat_vec_flat, k=real_k)
        # Compute the distance and store the original query
        score = self.__compute_distance(idx, dist, query, k, use_orig, optimized=selective, level2=level2)
        # print(score)
        # Whether use optimization approach -- selective overlapping
        if not selective:
            # Store the feature vector
            self.feat_vec_flats.append(feat_vec_flat)
        # Update the index
        self.index.add(feat_vec_flat)
        self.lsh_counter += 1
        return score


    def cal_perc_lsh_step1(self, query, k=K, use_orig=False,
            hnsw=False, level2=False):
        # Set k value based on level2
        real_k = k
        if level2:
            real_k = k * 5
        # feat_vec is a list of feature vector, the dimension depends on the model.
        feat_vec = self.loss_fn.feat(query.to(self.device))
        # Convert the list to one vector.
        feat_vec_flat = torch.cat(feat_vec, dim=1).cpu().numpy()
        # Initiate or reset the buffer
        if self.initial_flag:
            d = feat_vec_flat.shape[1]
            # Choose the index
            if hnsw:
                self.index = faiss.IndexHNSWFlat(d, self.M)
                self.index.hnsw.efConstruction = self.ef_construction
                self.index.hnsw.efSearch = self.ef_search
            else:
                self.index = faiss.IndexLSH(d, 2048)
            self.initial_flag = False
            self.lsh_counter = real_k
            self.index.add(np.zeros((real_k, d)).astype(np.float32))
            if use_orig:
                self.orig_queries = []
                for _ in range(real_k):
                    self.orig_queries.append(torch.zeros(query.shape).to(self.device))
        if self.lsh_counter == LSH_WINDOW_SIZE:
            self.index.remove_ids(np.array([0], dtype='int64'))
            self.lsh_counter -= 1
            self.orig_queries = self.orig_queries[1:]
        # Search in the index
        dist, idx = self.index.search(feat_vec_flat, k=real_k)
        # Compute the distance and store the original query
        score = self.__compute_distance(idx, dist, query, k, use_orig, level2=level2)
        # print(score)
        # Update the index
        self.index.add(feat_vec_flat)
        self.lsh_counter += 1
        return score


    # Cal hamming distance based on lsh
    def cal_ham_lsh(self, query, k=K, use_orig=False, hnsw=False, selective=False, level2=False):
        query_flat = query.reshape(1, -1).cpu().numpy()
        # Initiate or reset the buffer
        if self.initial_flag or self.lsh_counter == LSH_WINDOW_SIZE:
            d = query_flat.shape[1]
            self.__init_index(d, k, query.shape, use_orig, hnsw)
        # Search in the index
        real_k = k
        if level2:
            real_k = k * 5
        dist, idx = self.index.search(query_flat, k=real_k)
        # Compute the distance
        score = self.__compute_distance(idx, dist, query, k, use_orig, metric="l2", optimized=selective, level2=level2)
        # Update the index
        self.index.add(query_flat)
        if self.lsh_counter == LSH_WINDOW_SIZE - LSH_OVERLAP_SIZE:
            self.lsh_idx_reset = True
        if self.lsh_idx_reset:
            self.overlap_index.add(query_flat)
        self.lsh_counter += 1
        return score



def main():
    test_query = []
    fake_query = torch.rand((1, 3, 32, 32))
    device = "cuda:0"
    calDist = CalDist()
    for _ in range(110):
        fake_query = torch.rand((1, 3, 32, 32)).detach().to(device)
        test_query.append(fake_query)
        # result = calDist.cal_perc_lsh(fake_query, 5)
        # result = calDist.cal_perc_lsh(fake_query, 5, use_orig=True, hnsw=False)
        result = calDist.cal_perc_knn(fake_query)
        # result = calDist.cal_ham_lsh(fake_query, 5, use_orig=True, hnsw=True)
        # result = calDist.cal_perc_lsh_slide(fake_query, 5, hnsw=True, use_orig=True, selective=True, level2=True)
        # calDist.cal_perc_lsh_step1(fake_query, 5, use_orig=True)
        print(result)
    # result = calDist.cal_norm(fake_query, test_query, 5)
    # result = calDist.cal_perc(fake_query, test_query, 5)
    # print(result)
    return True

if __name__ == "__main__":
    main()
