#! python3
import os
import sys
import pickle
import json
import torch
import torchvision
import threading
import logging
import time

from torchvision import transforms
from pyflink.common.serialization import DeserializationSchema, SimpleStringSchema, JsonRowDeserializationSchema, SerializationSchema
from pyflink.common.typeinfo import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors import FlinkKafkaConsumer
from pyflink.datastream.functions import FlatMapFunction, ProcessFunction, MapFunction

# Log setting
logging.basicConfig(
  format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S',
  level=logging.INFO,
  handlers=[
      logging.FileHandler("query_processor.log"),
      logging.StreamHandler(sys.stdout)
  ]
)
logger = logging.getLogger()

# Some parameters
RATIO = 2
MAX_WINDOW = 10
K = 5

# Load benign queries
transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# Use test data as benign queries
benign_query = torchvision.datasets.CIFAR10(root='./data', train=False,
        download=True, transform=transform)
benign_loader = torch.utils.data.DataLoader(benign_query, batch_size=1,
        shuffle=True, num_workers=1)
benign_iter = iter(benign_loader)

# Insert benign queries
def insert_benign_query():
    global benign_iter
    for i in range(RATIO):
        try:
            image, _ = benign_iter.next()
            yield image[0]
        except StopIteration:
            benign_iter = iter(benign_loader)
            image, _ = benign_iter.next()
            yield image[0]

class Detect(ProcessFunction):
    def process_element(self, value, ctx: 'ProcessFunction.Context'):
        return super().process_element(value, ctx)


def main():
    query_buffer = []
    env = StreamExecutionEnvironment.get_execution_environment()
    kafka_jar = os.path.join(os.path.abspath(os.path.dirname(__file__)),
            'kafka/flink-sql-connector-kafka_2.11-1.14.3.jar')
    # the sql connector for kafka is used here as it's a fat jar and could avoid dependency issues
    env.add_jars("file://{}".format(kafka_jar))

    # deserialization_schema = JsonRowDeserializationSchema.builder() \
        # .type_info(type_info=Types.ROW([Types.INT(), Types.STRING()])).build()
    deserialization_schema = SimpleStringSchema()

    kafka_consumer = FlinkKafkaConsumer(
        topics='queries',
        deserialization_schema=deserialization_schema,
        properties={'bootstrap.servers': 'localhost:9092',
            'group.id': 'queries'})

    ds = env.add_source(kafka_consumer)

    def collector(query):
        logger.info("Query type: {}".format(type(query)))
        t_query = torch.Tensor(json.loads(query))
        query_buffer.append(t_query)
        # for b_query in insert_benign_query():
            # query_buffer.append(b_query)
        global benign_iter
        for i in range(RATIO):
            try:
                image, _ = benign_iter.next()
                query_buffer.append(image[0])
            except StopIteration:
                benign_iter = iter(benign_loader)
                image, _ = benign_iter.next()
                query_buffer.append(image[0])
        print("The size of query: {}".format(t_query.shape))
        if len(query_buffer) >= MAX_WINDOW:
            try:
                # threading.Thread(target=detect, args=(query_buffer, K))
                logger.info("The size of queries: {}".format(len(query_buffer)))
            except:
                logger.error("Thread init fails.")
        return 1

    def test_collector(query):
        return 1

    ds = ds.map(lambda a: collector(a), output_type=Types.INT())

    ds.print()

    env.execute("test_flink")


if __name__ == "__main__":
    main()
