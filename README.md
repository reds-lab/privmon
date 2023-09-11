# privmon
This is an official repository for PrivMon: A Stream-Based System for Real-Time Privacy Attack Detection for Machine Learning Models (RAID 2023)


Environment Setup for 

Conda env create -f environment.yml --name myenv

Replace lpips in anaconda package 

Download CelebA dataset:
https://www.google.com/search?q=celeba&rlz=1C5MACD_enUS1023US1024&oq=celeba&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIGCAEQRRhB0gEIMjE4OGowajeoAgCwAgA&sourceid=chrome&ie=UTF-8

Download models:
We use the same models. You can download them at link : https://drive.google.com/drive/folders/1U4gekn72UX_n1pHdm9GQUQwwYVDvpTfN
You can also download the generator model from link: https://drive.google.com/drive/folders/1L3frX-CE4j36pe5vVWuy3SgKGS9kkA70?usp=sharing.

Environment Setup for Privmon
1. Install Docker Engine
2. Start Docker service: 'sudo systemctl start docker'
3. Run: 'sudo docker-compose up'
4. Run: 'bash create topics.sh'

Decision-based 
2.1. Python main.py –action 0 [train model]
2.2. python main.py --blackadvattack HopSkipJump --dataset_ID 0 --datasets CIFAR10 -number_classes 10
2.3. python main.py -d CIFAR10 -a HSJ --metrics perc_lsh_step1_orig_level2

Membership inference 
2.1. Python training.py –ndata 1000 –dataset 'cifar10' [train model]
2.2. Python pipeline_main.py –ndata 1000 –dataset 'cifar10' –attacks 'r' –r 5 [attack]

Model inversion 
2.1. Python magnetic_main.py

Upload one malicious query + a benign one - from my side
Run the system on these queries. 
