# **PrivMon: Real-Time Privacy Attack Detection System**

This is the official repository for **PrivMon**, presented at RAID 2023. This system is a stream-based solution for real-time privacy attack detection targeting machine learning models.

## **Environment Setup**

1. **Conda Environment**
    ```bash
    conda env create -f environment.yml --name myenv
    ```

2. **Replace lpips in Anaconda Package**

## **System Setup for PrivMon**

1. **Docker Engine**
    - Install the Docker Engine from the official website.

2. **Docker Services**
    ```bash
    sudo systemctl start docker
    sudo docker-compose up
    ```

3. **Kafka Topics**
    ```bash
    bash create topics.sh
    ```

## **Simulate Attack and Detection**

### **Decision-Based Attacks**

1. **Train Model [in the corresponding attack folder]**
    ```bash
    python main.py –action 0
    ```

2. **Simulate the Attack [in the corresponding attack folder]**
    ```bash
    python main.py --blackadvattack HopSkipJump --dataset_ID 0 --datasets CIFAR10 --number_classes 10
    ```

3. **System Evaluation [in the ml-privacy folder]**
    ```bash
    python main.py -d CIFAR10 -a HSJ --metrics perc_lsh_step1_orig_level2
    ```
    

## **Model Inversion Attack Preparation**

1. **Datasets**
    - Download the **CelebA** and **Facescrub** datasets from their official websites and save these into the ".data" folder.

2. **Models**
    - We leverage the same target models and GANs as previous research. 
    - Download target models [here](https://drive.google.com/drive/folders/1U4gekn72UX_n1pHdm9GQUQwwYVDvpTfN).
    - Download generator [here](https://drive.google.com/drive/folders/1L3frX-CE4j36pe5vVWuy3SgKGS9kkA70?usp=sharing).
    - Save these models in the attack folder.
      
3. **Additional References**
    - For more details, please see the related project: [Label-Only Model Inversion Attacks via Boundary Repulsion](https://github.com/m-kahla/Label-Only-Model-Inversion-Attacks-via-Boundary-Repulsion).

## **Model Inversion Attack**

1. **Simulate the Attack [in the corresponding attack folder]**
    ```bash
    python magnetic_main.py
    ```
2. **System Evaluation [in the ml-privacy folder]**
    ```bash
    python main.py --dataset CelebA --metric perc_lsh_step1_orig_level2
    ```
