#! python3

# Some parameters
# The ratio of benign/malicious queries
RATIO = 1
# The tumbling window size
WINDOW_SIZE = 200
# The stored window numbers
MAX_WINDOW = 1000
# The window size of lsh
LSH_WINDOW_SIZE = 5000
# This one represent no sliding window
# LSH_WINDOW_SIZE = 5000000
# The overlapping window size of lsh
# This value should not be larger than window size / 2
LSH_OVERLAP_SIZE = 2000
# The K value of KNN
K = 2
# The size to report stream processing results in debug mode
STREAM_SIZE = 1000
# Dump size: The size to report results in experiment mode
# Also the number of images to save
DUMP_SIZE = 100*STREAM_SIZE

# High-likely malicious query threshold
LSH_MAL_THOLD = 0.03
# The topics of Kafka
TOPICS = ["queries"]

# Metric list
# The mse distance
# The perceptual distance
# The manhatton distance
# The lsh based on perceptual distance
# The lsh based on manhhaton distance
# The lsh based on original perceptual distance
# The lsh based on l2 distance
METRIC_LIST = ["mse", "perc", "manh", "perc_lsh", "ham_lsh", "perc_lsh_orig", "perc_lsh_slide", "perc_hnsw", "perc_hnsw_orig", "perc_lsh_orig_level2", "perc_hnsw_orig_level2", "perc_lsh_slide_orig", "perc_lsh_slide", "perc_lsh_slide_orig_level2", "perc_lsh_slide_orig_sel", "perc_lsh_slide_orig_level2_sel", "perc_lsh_step1_orig", "perc_lsh_step1_orig_level2", "perc_lsh_step1"]
# Metric description map
METRIC_DESC_MAP = {
        "mse": "MSE Distance",
        "perc": "Perceptual Distance",
        "manh": "Manhatton Distance",
        "perc_lsh": "Percptual LSH Distance",
        "hnsw_lsh": "Percptual HNSW Distance",
        "ham_lsh": "Hamming LSH Distance",
        "perc_lsh_orig": "Percuptual LSH Original Distance",
        "perc_lsh_orig_level2": "Percuptual LSH Original Distance Level2",
        "perc_hnsw_orig": "Percuptual HNSW Original Distance",
        "perc_hnsw_orig_level2": "Percuptual HNSW Original Distance Level2",
        "perc_lsh_slide": "Perc LSH Dist With Arbitrary Step Size",
        "perc_lsh_slide_orig": "Perc LSH Orig Dist With Arbitrary Step Size",
        "perc_lsh_slide_orig_level2": "Perc LSH Orig Dist Slide Level2",
        "perc_lsh_slide_orig_sel": "Perc LSH Orig Dist Slide Selective",
        "perc_lsh_slide_orig_level2_sel": "Perc LSH Orig Dist Slide Level2 Selective",
        "perc_lsh_step1_orig": "Perc LSH Orig Dist Step 1",
        "perc_lsh_step1_orig_level2": "Perc LSH Orig Dist Step 1 Level2",
        "perc_lsh_step1": "Perc LSH Dist Step 1",
        "perc_knn": "Perc KNN Without Sliding Window"
        }
# Metric threshold map
METRIC_THOLD_MAP = {
        "mse": 20,
        "perc": 0.7,
        "manh": 0,
        "perc_lsh": 600,
        "perc_hnsw": 600,
        "ham_lsh": 300,
        "perc_lsh_orig": 0.07,
        "perc_lsh_orig_level2": 0.07,
        "perc_hnsw_orig": 0.07,
        "perc_hnsw_orig_level2": 0.07,
        "perc_lsh_slide": 600,
        "perc_lsh_slide_orig": 0.07,
        "perc_lsh_slide_orig_level2": 0.07,
        "perc_lsh_slide_orig_sel": 0.07,
        "perc_lsh_slide_orig_level2_sel": 0.07,
        "perc_lsh_step1": 350, ##  400
        "perc_lsh_step1_orig": 0.07,
        "perc_lsh_step1_orig_level2": 0.07,
        # Stateful Detection et. (SPAI'20)
        "perc_knn": 1.44
        }
