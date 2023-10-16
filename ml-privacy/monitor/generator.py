#! python3
import argparse
import atexit
import time
import sys
import numpy as np
import torch
import pickle
import logging

from confluent_kafka import Producer
from tqdm import tqdm

from .config import TOPICS
from .utils import get_logger

# get logger
logger = get_logger(__name__, level=logging.DEBUG)

class ProducerCallback:
    def __init__(self, record, log_success=False):
        self.record = record
        self.log_success = log_success

    def __call__(self, err, msg):
        if err:
            logger.error('Error producing record {}'.format(self.record))
        elif self.log_success:
            pass
            # logger.info('Produced {} to topic {} partition {} offset {}'.format(
                # self.record,
                # msg.topic(),
                # msg.partition(),
                # msg.offset()
            # ))

class Feed:
    def __init__(self, bootstrap_server='localhost:9092'):
        logger.info('Starting query producer')
        self.counter = 0
        self.conf = {
                'bootstrap.servers': bootstrap_server,
                'linger.ms': 200,
                'client.id': TOPICS[0],
                'partitioner': 'murmur2_random'
                }
        self.producer = Producer(self.conf)

    def __del__(self):
        atexit.register(lambda p: p.flush(), self.producer)

    def feed2kafka(self, queries, data_type='torch'):
        # query.shape = (1, 3, 32, 32)
        if data_type == 'torch' and torch.is_tensor(queries):
            # print(queries.shape)
            # Copy from GPU to CPU
            queries = queries.cpu()
            # Separate the query batches
            query_inter = queries.split(1)
        elif data_type == 'numpy' and isinstance(queries, np.ndarray):
            query_inter = queries
        else:
            raise ValueError(f"data type: {data_type}, real type: {type(queries)}")
        for query in tqdm(query_inter):
            is_tenth = self.counter % 10 == 0
            # Value should be casted to numpy before transfer because the size
            # of pickle.dumps(queries) == pickle.dumps(query)
            if data_type == 'numpy':
                # Transform (32, 32, 3) to (3, 32 ,32)
                query = np.transpose(query, (2, 0, 1))
                value = pickle.dumps(np.expand_dims(query, axis=0))
            else:
                value = pickle.dumps(query.detach().numpy())
            # logger.debug("Size of the value: {}".format(len(value)))
            self.producer.produce(topic=TOPICS[0],
                            value=value,
                            on_delivery=ProducerCallback(query, log_success=is_tenth))
            # time.sleep(5)
            if is_tenth:
                # time.sleep(2)
                self.producer.poll(1)
                if self.counter % 100 == 0:
                    # time.sleep(1)
                    pass
            self.counter += 1




def main(args):
    logger.info('Starting query producer ...')
    feed = Feed(bootstrap_server=args.bootstrap_server)
    while True:
        # feed.feed2kafka(np.random.random((3,3)))
        feed.feed2kafka(torch.randn((100, 3, 32, 32)))
        time.sleep(100)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bootstrap-server', default='localhost:9092')
    parser.add_argument('--topic', default='queries')
    args = parser.parse_args()
    main(args)
