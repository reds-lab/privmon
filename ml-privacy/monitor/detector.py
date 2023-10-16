#! python3
import logging
import os
import numpy as np
import torch
import pprint
import time
import pickle
from functools import partial

from multiprocessing import Process, Pool
from .config import *
from .utils import get_logger
from .knn import CalDist

# Log setting
logger = get_logger(__name__, level=logging.DEBUG)


class Detector():
    def __init__(self, metrics, is_save=False,
            base_dir="data/results", device="cuda:0", dataset="CIFAR10"):
        self.metrics = metrics
        self.is_save = is_save
        # Base result dir
        self.base_dir = base_dir
        self.device = device
        self.dataset = dataset
        # Counter
        self.win_counter = 0
        # Distance calculator
        self.calDist = CalDist(device=device)
        # self.pool = Pool(pool_num)
        # Counters of stream processing
        self.stream_counter = 0
        self.stream_dist_label = []
        # Save images
        self.saved_images = []
        self.stream_time_record = time.time()
        # Time list
        self.time_list = []
        self.tp_counter = 0
        self.tn_counter = 0
        self.fp_counter = 0
        self.fn_counter = 0
        # the first detected malicious query
        self.is_first = True
        # if debug
        self.is_debug = False
        # some sizes
        if self.is_debug:
            self.print_size = STREAM_SIZE
        else:
            self.print_size = DUMP_SIZE
        self.save_size = DUMP_SIZE
        self.dump_size = DUMP_SIZE
        self.metricMap = {
                'mse': partial(self.calDist.cal_norm, ord=2),
                'manh': partial(self.calDist.cal_norm, ord=1),
                'ham': self.calDist.cal_ham,
                'perc': self.calDist.cal_perc
                }
        self.streamMap = {
                'perc_lsh': partial(self.calDist.cal_perc_lsh, use_orig=False),
                'perc_lsh_orig': partial(self.calDist.cal_perc_lsh, use_orig=True),
                'perc_lsh_orig_level2': partial(self.calDist.cal_perc_lsh, use_orig=True, level2=True),
                'perc_hnsw': partial(self.calDist.cal_perc_lsh, use_orig=False, hnsw=True),
                'perc_hnsw_orig': partial(self.calDist.cal_perc_lsh, use_orig=True, hnsw=True),
                'perc_hnsw_orig_level2': partial(self.calDist.cal_perc_lsh, use_orig=True, hnsw=True, level2=True),
                'perc_lsh_slide': partial(self.calDist.cal_perc_lsh_slide, use_orig=False),
                'perc_lsh_slide_sel': partial(self.calDist.cal_perc_lsh_slide, use_orig=False, selective=True),
                'perc_lsh_slide_orig': partial(self.calDist.cal_perc_lsh_slide, use_orig=True),
                'perc_lsh_slide_orig_level2': partial(self.calDist.cal_perc_lsh_slide, use_orig=True, level2=True),
                'perc_lsh_slide_orig_sel': partial(self.calDist.cal_perc_lsh_slide, use_orig=True, selective=True),
                'perc_lsh_slide_orig_level2_sel': partial(self.calDist.cal_perc_lsh_slide, use_orig=True, level2=True, selective=True),
                'perc_lsh_step1': partial(self.calDist.cal_perc_lsh_step1, use_orig=False),
                'perc_lsh_step1_orig': partial(self.calDist.cal_perc_lsh_step1, use_orig=True),
                'perc_lsh_step1_orig_level2': partial(self.calDist.cal_perc_lsh_step1, use_orig=True, level2=True),
                'ham_lsh': self.calDist.cal_ham_lsh,
                'perc_knn': self.calDist.cal_perc_knn_v2
                }

    def stop_pool(self):
        # self.pool.close()
        # self.pool.join()
        return True

    def cal_batch_dist(self, queries, metric='mse'):
        dists = []
        stime = time.time()
        if metric in self.metricMap.keys():
            dist_fun = self.metricMap[metric]
            for query in queries:
                d = dist_fun(query, queries, K)
                dists.append(d)
        # Test based on lsh
        elif metric in self.streamMap.keys():
            dist_fun = self.streamMap[metric]
            for query in queries:
                d = dist_fun(query, K)
                dists.append(d)
        else:
            logger.warning("Metric does not exists!")
            return -1, -1
        etime = time.time()
        return dists, etime-stime

    def analy_result(self, dist_label, thold):
        positive = 1
        negative = 0
        total_num = dist_label.shape[0]
        results = dist_label[:,0]
        labels = dist_label[:,1]
        results = np.where(results>thold, 0, 1)
        tp = np.sum(np.logical_and(results == positive, labels == positive))
        tn = np.sum(np.logical_and(results == negative, labels == negative))
        fp = np.sum(np.logical_and(results == positive, labels == negative))
        fn = np.sum(np.logical_and(results == negative, labels == positive))
        return tp/(tp+fp), tp/(tp+fn), (tp+tn)/total_num

    def save_result(self, result, path):
        dirs = os.path.dirname(path)
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        with open(path, 'wb') as fp:
            pickle.dump(result, fp)
        print(f"{path} is saved.")

    def batch_detect(self, queries, labels):
        self.win_counter += 1
        np_labels = np.array(labels)
        for metric in self.metrics:
            # Validate metric
            if metric not in METRIC_LIST:
                logger.warning(f"Metric {metric} Does Not Exist in Batch Processing!")
                pass
            # Calculate the distance
            dist, time = self.cal_batch_dist(queries, metric=metric)
            dist_label = np.concatenate((np.array(dist).reshape(-1,1), np_labels.reshape(-1, 1)), axis=1)
            # print(dist_label)
            if self.is_save:
                logger.debug("Saving the {}th result".format(self.win_counter))
                base_path = os.path.join(self.base_dir, f"/{self.dataset}/{metric}/w{WINDOW_SIZE}-k{K}")
                self.save_result(dist_label, base_path + f"/n{self.win_counter}.pickle")
            # Ayalyze the result
            precision, recall, accu = self.analy_result(dist_label, METRIC_THOLD_MAP[metric])
            logger.debug(f">>>>>>>>> RESULT of {METRIC_DESC_MAP[metric]} with {WINDOW_SIZE} queries:")
            logger.debug(f"Time: {time}, Precision: {precision}, Recall: {recall}, Accuracy: {accu}")
        # Store up to 100 windows 
        if self.is_save and self.win_counter == MAX_WINDOW:
            return False
        return True

    def stream_detect(self, query, label):
        if len(self.metrics) == 0:
            logger.warning("No metric!")
            return False
        metric = self.metrics[0]
        if metric not in self.streamMap.keys():
            logger.warning(f"Metric {metric} does not exist in stream processing!")
            return False
        dist_fun = self.streamMap[metric]
        # calculate the distance
        try:
            dist = dist_fun(query)
        except Exception as e:
            logger.error(f"Exception: {e}")
            return False
        # The first k queries
        if dist == -1:
            return True
        self.stream_dist_label.append([dist, label])
        res_label = 0 if dist > METRIC_THOLD_MAP[metric] else 1
        # print(dist)
        if label==1 and res_label==1:
            self.tp_counter += 1
        elif label==0 and res_label==0:
            self.tn_counter += 1
        elif label==1 and res_label==0:
            self.fn_counter += 1
        else:
            self.fp_counter += 1
        self.stream_counter += 1
        save_images = False
        if save_images:
            self.saved_images.append([query, label])
        # Print the first malicious query
        if self.is_first and self.tp_counter==1:
            self.is_first = False
            logger.info("==============================")
            logger.info(f"The first detected malicious query: {self.stream_counter}")
            logger.info("==============================")
        # Print information
        if self.stream_counter % self.print_size == 0:
            self.print_info(metric)
        # Dumping images
        if save_images and self.stream_counter % self.dump_size == 0:
            self.dump_image(metric)
        # Max stream size
        if self.is_save and self.stream_counter == MAX_WINDOW * self.print_size:
            logger.info(f"Average Time: {np.mean(self.time_list)}")
            return -1
        return res_label


    def dump_image(self, metric):
        base_path = os.path.join(self.base_dir, f"{self.dataset}")
        file_path = base_path + f"/images{self.stream_counter//self.dump_size}.pickle"
        logger.info("==============================")
        logger.info(f"Dumping {self.dump_size} images into {file_path}")
        logger.info("==============================")
        self.save_result(self.saved_images, file_path)
        self.saved_images = []


    def print_info(self, metric):
        self.stream_report(self.print_size)
        if self.is_save and self.stream_counter % self.save_size == 0:
            logger.debug(f"Saving stream processing results")
            dist_label = np.array(self.stream_dist_label)
            base_path = os.path.join(self.base_dir, f"{self.dataset}/{metric}/w{LSH_WINDOW_SIZE}-o{LSH_OVERLAP_SIZE}-k{K}")
            if self.is_debug:
                self.save_result(dist_label, base_path + f"/n{self.stream_counter//self.save_size}.pickle")
            else:
                self.save_result(dist_label, base_path + f"-result{self.stream_counter//self.save_size}.pickle")
        self.stream_dist_label = []

    def stream_report(self, total_num):
        tp = self.tp_counter / total_num
        tn = self.tn_counter / total_num
        fp = self.fp_counter / total_num
        fn = self.fn_counter / total_num
        self.tp_counter = 0
        self.tn_counter = 0
        self.fp_counter = 0
        self.fn_counter = 0
        logger.debug(f">>>>>>>> The {self.stream_counter//self.print_size}th Stream Processing Result:")
        current_time = time.time()
        logger.debug(f"Time of {self.print_size} queries: {current_time-self.stream_time_record}")
        self.time_list.append(current_time-self.stream_time_record)
        self.stream_time_record = current_time
        if tp+fp>0 and tp+fn>0:
            logger.debug(f"TP: {tp}, TN : {tn}, FP: {fp}, FN: {fn}, total : {tp+tn+fp+fn}")        
            logger.debug(f"Precision: {tp/(tp+fp)}, Recall: {tp/(tp+fn)}, Accuracy: {tp+tn}")
        else:
            pprint.pprint(self.stream_dist_label)
            logger.debug(f"Precision: -1, Recall: -1, Accuracy: -1")
        return tp, tn, fp, fn

    # bugs
    def multiprocess_detect(self, queries, labels):
        # res = self.pool.apply_async(func=self.detect, args=(queries, labels))
        # print(res.get(timeout=1))
        return True


def main():
    # logger = get_logger(__name__)
    test_query = []
    test_label = []
    device = torch.device("cuda:0")
    for _ in range(WINDOW_SIZE):
        fake_query = torch.rand((1, 3, 32, 32)).to(device)
        fake_label = 1
        test_query.append(fake_query)
        test_label.append(fake_label)

    detector = Detector(metrics=["ham_lsh"])
    logger.info("Starting detector ...")
    # detector.batch_detect(test_query, test_label)
    for i in range(len(test_query)):
        detector.stream_detect(test_query[i], test_label[i])
    logger.info("Detector end!")
    # detector.stop_pool()
    logger.info("Return!")
    return True

if __name__ == "__main__":
    main()
