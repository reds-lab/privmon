#! python3
import os, sys
import torch
import argparse
from confluent_kafka import Consumer
from monitor.processor import Processor, main_process
from monitor.utils import get_logger, kafka_consumer_conf
from monitor.config import *

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Query Stream Processing') 
    parser.add_argument('--exhaust', '-e', default=False,
                        action="store_const", const=True,
                        help='Process or exhaust the remaining queries in kafka buffer')
    parser.add_argument('--cpu', '-c', default=False,
                        action="store_const", const=True,
                        help='Whether to use CPU to not')
    parser.add_argument('--dataset', '-d', default="CIFAR10", type=str,
                        help='The benign dataset')
    parser.add_argument('--metrics', '-m', default=["mse", "perc"],
                        type=str, nargs="+",
                        help=f'The metric list that will be used to evaluate. Available metrics {METRIC_LIST}')
    parser.add_argument('--save','-s', default=False,
                        action="store_const", const=True,
                        help='Whether save the results or not')
    parser.add_argument('--attack', '-a', default="default", type=str,
                        help='The attack and the saving path')
    parser.add_argument('--processtype', '-p', default="stream", type=str,
                        choices=["stream", "batch"],
                        help='The processing type')
    parser.add_argument('--offline', default="None", type=str,
                        help='Using offline data')
    args = parser.parse_args()
    # print(args)
    if args.cpu:
        device = "cpu"
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.processtype == "stream":
        log_filepath = f"data/{args.attack}/{args.dataset}-{args.metrics[0]}-w{LSH_WINDOW_SIZE}-o{LSH_OVERLAP_SIZE}-k{K}"
    else:
        log_filepath = f"data/{args.attack}/{args.dataset}-{args.metrics[0]}-w{WINDOW_SIZE}-k{K}"
    if not os.path.exists(os.path.dirname(log_filepath)):
        os.makedirs(os.path.dirname(log_filepath))
    logger = get_logger(__name__, fname=log_filepath)

    logger.info("=================================================")
    logger.info(f"Saving log on file {log_filepath}")
    logger.info(f"The arguments: {args}")
    logger.info(f"The ratio is: {RATIO}")
    logger.info(f"The window size is: {WINDOW_SIZE}")
    logger.info(f"The max window is: {MAX_WINDOW}")
    logger.info(f"The lsh window size is: {LSH_WINDOW_SIZE}")
    logger.info(f"The lsh overlapping window size is: {LSH_OVERLAP_SIZE}")
    logger.info(f"The size to report stream processing results: {STREAM_SIZE}")
    logger.info(f"The K value is: {K}")
    logger.info(f"Using offline data: {args.offline}")
    logger.info("=================================================")

    logger.info("Processor starting ...")
    if args.offline == "None":
        consumer = Consumer(kafka_consumer_conf)
    else:
        consumer = None
    process = Processor(args.dataset, args.metrics, args.save, args.processtype, "data/"+args.attack, device)
    main_process(consumer, process, args.exhaust, args.offline)


if __name__ == "__main__":
    main()
