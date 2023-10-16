#! python3
import sys
import pickle
import numpy as np
import torch
import time
import argparse
import logging
from torch.utils import data
from tqdm import tqdm

from confluent_kafka import Consumer, KafkaError, KafkaException
from .config import *
from .utils import get_logger, load_samples, kafka_consumer_conf
from .detector import Detector

# Log setting
logger = get_logger(__name__, level=logging.DEBUG)

class Processor:
    def __init__(self, dataset, metrics, is_save=False,
            process_type = "stream",
            basedir="data/results", device="cuda:0"):
        self.dataset = dataset
        self.device = device
        if process_type == "stream":
            logger.info("Starting Stream Processing ...")
            self.process_fun = self.stream_process
        elif process_type == "batch":
            logger.info("Starting Batch Processing ...")
            self.process_fun = self.batch_process
        else:
            raise ValueError(f"The process type {process_type} does not exist")
        # Query buffer
        self.query_buffer = []
        self.label_buffer = []
        self.detector = Detector(metrics, is_save, basedir, device, dataset)
        self.train_iter = load_samples(self.dataset, True)
        # self.test_iter = load_samples(self.dataset, False)

    # Insert benign queries
    def gen_benign_query(self):
        for _ in range(RATIO):
            try:
                train_image, _ = self.train_iter.next()
            except StopIteration:
                self.train_iter = load_samples(self.dataset, True)
                train_image, _ = self.train_iter.next()
            # try:
                # test_image, _ = self.test_iter.next()
            # except StopIteration:
                # self.test_iter = load_samples(self.dataset, False)
                # test_image, _ = self.test_iter.next()
            # yield (train_image + test_image) / 2
            yield train_image

    # collect malicious (1, query) and benign (0, query) queries
    # query.shape = (1, 3, 32, 32)
    def batch_process(self, m_query):
        # print(m_query.shape)
        self.query_buffer.append(m_query.detach().to(self.device))
        self.label_buffer.append(1)
        for b_query in self.gen_benign_query():
            self.query_buffer.append(b_query.detach().to(self.device))
            self.label_buffer.append(0)
        # logger.debug("The size of query: {}".format(m_query.shape))
        if len(self.query_buffer) >= WINDOW_SIZE:
            try:
                # Call detector
                # logger.debug("The number of queries: {}".format(len(self.query_buffer)))
                # stime = time.time()
                flag = self.detector.batch_detect(self.query_buffer, self.label_buffer)
                if not flag: shutdown()
                # print(">>>>>>>>>> Processing time: {}".format(time.time()-stime))
                # self.detector.multiprocess_detect(self.query_buffer, self.label_buffer)
                self.query_buffer = []
                self.label_buffer = []
            except Exception as ex:
                logger.error(ex.args)
        return True

    # collect malicious (1, query) and benign (0, query) queries
    def stream_process(self, m_query):
        return_map = {0: False, 1: True, -1: False}
        # process malicious queries
        flag = self.detector.stream_detect(m_query.detach().to(self.device), 1)
        if flag == -1: shutdown()
        return_value = return_map[flag]
        # process benign queries
        for b_query in self.gen_benign_query():
            flag = self.detector.stream_detect(b_query.detach().to(self.device), 0)
            if flag == -1: shutdown()
        # logger.debug("The size of query: {}".format(m_query.shape))
        return return_value

    def process(self, query):
        return self.process_fun(query)

    def defender(self, queries, data_type='torch'):
        # query.shape = (1, 3, 32, 32)
        if data_type == 'torch' and torch.is_tensor(queries):
            # Copy from GPU to CPU
            queries = queries.cpu()
        elif data_type == 'numpy' and isinstance(queries, np.ndarray):
            queries = queries
        else:
            raise ValueError(f"data type: {data_type}, real type: {type(queries)}")
        for idx, query in tqdm(enumerate(queries)):
            # Value should be casted to numpy before transfer because the size
            # of pickle.dumps(queries) == pickle.dumps(query)
            if data_type == 'numpy':
                # Transform (32, 32, 3) to (3, 32 ,32)
                query = np.transpose(query, (2, 0, 1))
            else:
                query = query.detach().numpy()
            # Expand from (3, 32, 32) to (1, 3, 32, 32)
            query = np.expand_dims(query, axis=0)

            result = self.stream_process(query)
            if result == True:
                random_query = np.random.random(query[0].shape)
                if data_type == 'numpy':
                    random_query = np.transpose(random_query, (1, 2, 0))
                queries[idx] = random_query
        return queries


# Online Processing
# Kafka consumer loop
running = True
def main_consume_loop(consumer, process, is_exhaust=False):
    try:
        consumer.subscribe(TOPICS)
        while running:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    # End of partition event
                    sys.stderr.write('%% %s [%d] reached end at offset %d\n' %
                                     (msg.topic(), msg.partition(), msg.offset()))
                elif msg.error():
                    raise KafkaException(msg.error())
            elif is_exhaust:
                continue
            else:
                try:
                    loaded_data = pickle.loads(msg.value())
                    if isinstance(loaded_data, np.ndarray):
                        m_query = torch.from_numpy(loaded_data)
                        process.process(m_query)
                        # logger.debug("Value len: {}".format(msg.len()))
                except pickle.PickleError as pex:
                    logger.error(f"Pickle Exception: {pex}")
    except Exception as ex:
        logger.error(f"Kafka Exception : {ex}")
    finally:
        # Close down consumer to commit final offsets.
        consumer.close()

def shutdown():
    logger.info("Stopping the process")
    global running
    running = False
    return True

# Offline Processing
def load_and_process(process, dataset):
    try:
        with open(dataset, "rb") as fo:
            datasource = pickle.load(fo)
            for (query,label) in tqdm(datasource):
                process.detector.stream_detect(query, label)
    except Exception as ex:
        logger.error(f"Offline Processing Exception : {ex}")


def main_process(consumer, process, is_exhaust, dataset):
    if dataset == "None":
        main_consume_loop(consumer, process, is_exhaust)
    else:
        load_and_process(process, dataset)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Query Stream Processing') 
    parser.add_argument('--exhaust', default=False, type=bool,
                        help='Process or Exhaust')
    parser.add_argument('--dataset', default="CIFAR10", type=str,
                        help='The benign dataset')
    args = parser.parse_args()
    logger.info("Processor starting ...")
    consumer = Consumer(kafka_consumer_conf)
    process = Processor(dataset=args.dataset, metrics=["mse", "perc"])
    main_consume_loop(consumer, process, args.exhaust)


if __name__ == "__main__":
    main()
