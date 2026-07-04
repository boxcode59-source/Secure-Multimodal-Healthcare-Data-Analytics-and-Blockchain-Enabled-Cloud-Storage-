"""
==============================================================
SPFI-HC
Smart Privacy-Preserving Federated Intelligence Framework
Part 1A-1
Configuration File
==============================================================
"""

import os
import random
import warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")

###############################################
# RANDOM SEED
###############################################

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


####################################################
# DEVICE
####################################################

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Running on:", DEVICE)


####################################################
# PROJECT ROOT
####################################################

PROJECT_ROOT = os.getcwd()

DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets")

IMAGE_DIR = os.path.join(DATASET_DIR, "images")

CSV_DIR = os.path.join(DATASET_DIR, "csv")

IOT_DIR = os.path.join(DATASET_DIR, "iot")

TEXT_DIR = os.path.join(DATASET_DIR, "text")

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

RESULT_DIR = os.path.join(PROJECT_ROOT, "results")

GRAPH_DIR = os.path.join(PROJECT_ROOT, "graphs")

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")


####################################################
# CREATE DIRECTORIES
####################################################

dirs = [

    DATASET_DIR,

    IMAGE_DIR,

    CSV_DIR,

    IOT_DIR,

    TEXT_DIR,

    CHECKPOINT_DIR,

    RESULT_DIR,

    GRAPH_DIR,

    LOG_DIR

]

for d in dirs:

    os.makedirs(d, exist_ok=True)


####################################################
# IMAGE SETTINGS
####################################################

IMAGE_SIZE = 224

CHANNELS = 3


####################################################
# TEXT SETTINGS
####################################################

MAX_TEXT_LENGTH = 256

VOCAB_SIZE = 50000

EMBED_DIM = 768


####################################################
# SIGNAL SETTINGS
####################################################

SIGNAL_LENGTH = 256

SIGNAL_CHANNELS = 1


####################################################
# TABULAR SETTINGS
####################################################

NUMERIC_FEATURES = 64

CATEGORICAL_FEATURES = 24


####################################################
# MODEL PARAMETERS
####################################################

NUM_CLASSES = 5

BATCH_SIZE = 16

EPOCHS = 100

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

MOMENTUM = 0.9

DROPOUT = 0.30

HIDDEN_DIM = 512

NUM_HEADS = 8

NUM_LAYERS = 4

MLP_DIM = 1024

CAPSULE_DIM = 32

ROUTING_ITERATIONS = 3

RNN_HIDDEN = 256

RNN_LAYERS = 2


####################################################
# FEDERATED PARAMETERS
####################################################

NUM_CLIENTS = 5

LOCAL_EPOCHS = 5

GLOBAL_ROUNDS = 100

CLIENT_FRACTION = 1.0

SCAFFOLD_MU = 0.01

SERVER_LR = 1.0


####################################################
# BLOCKCHAIN PARAMETERS
####################################################

BLOCK_SIZE = 32

HASH_TYPE = "sha256"

AES_KEY_SIZE = 32

CHAIN_LENGTH = 1000


####################################################
# ENCRYPTION
####################################################

AES_MODE = "CBC"

IV_SIZE = 16

KEY_SIZE = 32


####################################################
# TRAIN SPLIT
####################################################

TRAIN_RATIO = 0.70

VAL_RATIO = 0.15

TEST_RATIO = 0.15


####################################################
# EARLY STOPPING
####################################################

PATIENCE = 15

MIN_DELTA = 0.0001


####################################################
# SAVE FILES
####################################################

BEST_MODEL = os.path.join(

    CHECKPOINT_DIR,

    "best_model.pth"

)

LAST_MODEL = os.path.join(

    CHECKPOINT_DIR,

    "last_model.pth"

)

MODEL_HISTORY = os.path.join(

    RESULT_DIR,

    "history.csv"

)

METRIC_FILE = os.path.join(

    RESULT_DIR,

    "metrics.xlsx"

)


####################################################
# IMAGE TRANSFORM
####################################################

MEAN = [

    0.485,

    0.456,

    0.406

]

STD = [

    0.229,

    0.224,

    0.225

]


####################################################
# PRINT CONFIG
####################################################

print("=" * 60)

print("SPFI-HC CONFIGURATION")

print("=" * 60)

print("Epochs :", EPOCHS)

print("Batch Size :", BATCH_SIZE)

print("Learning Rate :", LEARNING_RATE)

print("Clients :", NUM_CLIENTS)

print("Classes :", NUM_CLASSES)

print("Image Size :", IMAGE_SIZE)

print("Device :", DEVICE)

print("=" * 60)
"""
==========================================================
SPFI-HC
Part 1A-2
Multimodal Dataset Loader
==========================================================
"""

import os
import glob
import random
import numpy as np
import pandas as pd

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from torchvision import transforms

from config import *


###########################################################
# IMAGE TRANSFORM
###########################################################

train_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ToTensor(),

    transforms.Normalize(MEAN, STD)

])



test_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(MEAN, STD)

])


###########################################################
# IMAGE FILES
###########################################################

def load_image_paths(image_root):

    image_paths = []

    labels = []

    classes = sorted(os.listdir(image_root))

    for cls in classes:

        folder = os.path.join(image_root, cls)

        if not os.path.isdir(folder):
            continue

        imgs = glob.glob(folder + "/*.png")

        imgs += glob.glob(folder + "/*.jpg")

        imgs += glob.glob(folder + "/*.jpeg")

        for img in imgs:

            image_paths.append(img)

            labels.append(cls)

    return image_paths, labels


###########################################################
# EHR CSV
###########################################################

def load_ehr(csv_file):

    df = pd.read_csv(csv_file)

    return df


###########################################################
# IoT DATA
###########################################################

def load_iot(csv_file):

    df = pd.read_csv(csv_file)

    return df


###########################################################
# TEXT DATA
###########################################################

def load_text(csv_file):

    df = pd.read_csv(csv_file)

    return df


###########################################################
# LABEL ENCODER
###########################################################

encoder = LabelEncoder()


###########################################################
# DATASET
###########################################################

class SPFIHealthcareDataset(Dataset):

    def __init__(

            self,

            image_paths,

            labels,

            ehr,

            iot,

            clinical_text,

            transform=None

    ):

        self.image_paths = image_paths

        self.labels = encoder.fit_transform(labels)

        self.ehr = ehr

        self.iot = iot

        self.text = clinical_text

        self.transform = transform

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, idx):

        #######################################
        # IMAGE
        #######################################

        img = Image.open(

            self.image_paths[idx]

        ).convert("RGB")

        if self.transform:

            img = self.transform(img)

        #######################################
        # TABULAR
        #######################################

        ehr_row = self.ehr.iloc[

            idx % len(self.ehr)

        ].values.astype(np.float32)

        #######################################
        # IOT
        #######################################

        iot_row = self.iot.iloc[

            idx % len(self.iot)

        ].values.astype(np.float32)

        #######################################
        # TEXT
        #######################################

        text = str(

            self.text.iloc[

                idx % len(self.text)

            ][0]

        )

        #######################################
        # LABEL
        #######################################

        label = self.labels[idx]

        return {

            "image": img,

            "ehr": torch.tensor(ehr_row),

            "iot": torch.tensor(iot_row),

            "text": text,

            "label": torch.tensor(label)

        }


###########################################################
# SPLIT
###########################################################

def create_dataset():

    image_paths, labels = load_image_paths(

        IMAGE_DIR

    )

    ehr = load_ehr(

        os.path.join(CSV_DIR, "ehr.csv")

    )

    iot = load_iot(

        os.path.join(IOT_DIR, "iot.csv")

    )

    clinical = load_text(

        os.path.join(TEXT_DIR, "clinical_text.csv")

    )

    train_imgs, test_imgs, train_lbls, test_lbls = train_test_split(

        image_paths,

        labels,

        test_size=0.30,

        random_state=SEED,

        stratify=labels

    )

    val_imgs, test_imgs, val_lbls, test_lbls = train_test_split(

        test_imgs,

        test_lbls,

        test_size=0.50,

        random_state=SEED,

        stratify=test_lbls

    )

    train_dataset = SPFIHealthcareDataset(

        train_imgs,

        train_lbls,

        ehr,

        iot,

        clinical,

        transform=train_transform

    )

    val_dataset = SPFIHealthcareDataset(

        val_imgs,

        val_lbls,

        ehr,

        iot,

        clinical,

        transform=test_transform

    )

    test_dataset = SPFIHealthcareDataset(

        test_imgs,

        test_lbls,

        ehr,

        iot,

        clinical,

        transform=test_transform

    )

    return train_dataset, val_dataset, test_dataset


###########################################################
# DATALOADER
###########################################################

def create_dataloader():

    train_dataset, val_dataset, test_dataset = create_dataset()

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=4,

        pin_memory=True

    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=4

    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=4

    )

    return train_loader, val_loader, test_loader


###########################################################
# TEST
###########################################################

if __name__ == "__main__":

    train_loader, val_loader, test_loader = create_dataloader()

    print("Train Samples :", len(train_loader.dataset))
    print("Validation Samples :", len(val_loader.dataset))
    print("Test Samples :", len(test_loader.dataset))

    sample = next(iter(train_loader))

    print(sample["image"].shape)
    print(sample["ehr"].shape)
    print(sample["iot"].shape)
    print(sample["label"].shape)
"""
===========================================================
SPFI-HC
Part 1A-3
Preprocessing Utilities
===========================================================
"""

import numpy as np
import pandas as pd
import torch

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

##########################################################
# DATA VALIDATION
##########################################################

class DataValidator:

    def __init__(self):
        pass

    def check_missing(self, dataframe):

        missing = dataframe.isnull().sum()

        print("\nMissing Values")

        print(missing)

        return missing

    def check_duplicate(self, dataframe):

        duplicates = dataframe.duplicated().sum()

        print("\nDuplicate Samples :", duplicates)

        return duplicates

    def check_datatype(self, dataframe):

        print("\nColumn Types")

        print(dataframe.dtypes)

    def summary(self, dataframe):

        print(dataframe.describe())


##########################################################
# MISSING VALUE IMPUTATION
##########################################################

class MissingValueHandler:

    def __init__(self):

        self.imputer = SimpleImputer(

            strategy="mean"

        )

    def fit(self, x):

        self.imputer.fit(x)

    def transform(self, x):

        return self.imputer.transform(x)

    def fit_transform(self, x):

        return self.imputer.fit_transform(x)


##########################################################
# STANDARD SCALER
##########################################################

class StandardNormalization:

    def __init__(self):

        self.scaler = StandardScaler()

    def fit(self, x):

        self.scaler.fit(x)

    def transform(self, x):

        return self.scaler.transform(x)

    def fit_transform(self, x):

        return self.scaler.fit_transform(x)


##########################################################
# MINMAX SCALER
##########################################################

class MinMaxNormalization:

    def __init__(self):

        self.scaler = MinMaxScaler()

    def fit(self, x):

        self.scaler.fit(x)

    def transform(self, x):

        return self.scaler.transform(x)

    def fit_transform(self, x):

        return self.scaler.fit_transform(x)


##########################################################
# ROBUST SCALER
##########################################################

class RobustNormalization:

    def __init__(self):

        self.scaler = RobustScaler()

    def fit(self, x):

        self.scaler.fit(x)

    def transform(self, x):

        return self.scaler.transform(x)

    def fit_transform(self, x):

        return self.scaler.fit_transform(x)


##########################################################
# QUANTILE ESTIMATION
##########################################################

class QuantileEstimator:

    def __init__(self, lower=5, upper=95):

        self.lower = lower

        self.upper = upper

    def compute(self, x):

        q_low = np.percentile(

            x,

            self.lower,

            axis=0

        )

        q_high = np.percentile(

            x,

            self.upper,

            axis=0

        )

        return q_low, q_high


##########################################################
# FEATURE ALIGNMENT
##########################################################

class FeatureAlignment:

    def __init__(self):

        pass

    def align(self, *features):

        min_length = min(

            feature.shape[0]

            for feature in features

        )

        aligned = [

            feature[:min_length]

            for feature in features

        ]

        return aligned


##########################################################
# NUMPY TO TENSOR
##########################################################

class TensorConverter:

    def __init__(self):

        pass

    def convert(self, array):

        return torch.tensor(

            array,

            dtype=torch.float32

        )


##########################################################
# DATA PIPELINE
##########################################################

class DataPipeline:

    def __init__(self):

        self.imputer = MissingValueHandler()

        self.scaler = StandardNormalization()

        self.quantile = QuantileEstimator()

    def process(self, dataframe):

        x = dataframe.values

        x = self.imputer.fit_transform(x)

        x = self.scaler.fit_transform(x)

        q1, q2 = self.quantile.compute(x)

        return x, q1, q2


##########################################################
# TEST
##########################################################

if __name__ == "__main__":

    df = pd.DataFrame(

        np.random.rand(100,10)

    )

    validator = DataValidator()

    validator.summary(df)

    pipeline = DataPipeline()

    x,q1,q2 = pipeline.process(df)

    print(x.shape)

    print(q1.shape)

    print(q2.shape)

"""
=========================================================
SPFI-HC
Part 2A

Quantile Self-Adaptive Min-Max Normalization (QSAMMN)

Equation (1)
=========================================================
"""

import numpy as np
import pandas as pd


class QuantileSelfAdaptiveMinMaxNormalization:

    """
    Proposed QSAMMN Normalization

    Equation (1)

    X_norm =
    (X-Q5)/(Q95-Q5+epsilon)

    """

    def __init__(self,
                 lower_quantile=5,
                 upper_quantile=95,
                 epsilon=1e-8):

        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.epsilon = epsilon

        self.q_low = None
        self.q_high = None

    ##########################################################

    def fit(self, X):

        X = np.asarray(X).astype(np.float32)

        self.q_low = np.percentile(
            X,
            self.lower_quantile,
            axis=0
        )

        self.q_high = np.percentile(
            X,
            self.upper_quantile,
            axis=0
        )

    ##########################################################

    def transform(self, X):

        X = np.asarray(X).astype(np.float32)

        X = (

            X - self.q_low

        ) / (

            self.q_high
            - self.q_low
            + self.epsilon

        )

        X = np.clip(X, 0, 1)

        return X

    ##########################################################

    def fit_transform(self, X):

        self.fit(X)

        return self.transform(X)


#############################################################
# Data Statistics
#############################################################

class DatasetStatistics:

    def __init__(self):
        pass

    def statistics(self, X):

        X = np.asarray(X)

        stats = {

            "Samples": X.shape[0],

            "Features": X.shape[1],

            "Mean": np.mean(X),

            "Std": np.std(X),

            "Minimum": np.min(X),

            "Maximum": np.max(X),

            "Median": np.median(X)

        }

        return stats


#############################################################
# Adaptive Range Analyzer
#############################################################

class AdaptiveRangeAnalyzer:

    def __init__(self):

        pass

    def analyze(self, X):

        X = np.asarray(X)

        report = []

        for i in range(X.shape[1]):

            feature = X[:, i]

            report.append({

                "feature": i,

                "minimum": np.min(feature),

                "maximum": np.max(feature),

                "range": np.max(feature) - np.min(feature),

                "mean": np.mean(feature)

            })

        return pd.DataFrame(report)


#############################################################
# Quantile Report
#############################################################

class QuantileReport:

    def __init__(self):

        pass

    def generate(self, X):

        report = {

            "Q5": np.percentile(X, 5, axis=0),

            "Q25": np.percentile(X, 25, axis=0),

            "Q50": np.percentile(X, 50, axis=0),

            "Q75": np.percentile(X, 75, axis=0),

            "Q95": np.percentile(X, 95, axis=0)

        }

        return report


#############################################################
# QSAMMN Pipeline
#############################################################

class QSAMMN:

    """
    Complete QSAMMN Module

    Stage 1

    Statistics

    Stage 2

    Quantile Estimation

    Stage 3

    Adaptive Normalization

    """

    def __init__(self):

        self.stats = DatasetStatistics()

        self.quantile = QuantileReport()

        self.normalizer = QuantileSelfAdaptiveMinMaxNormalization()

    def process(self, X):

        statistics = self.stats.statistics(X)

        quantiles = self.quantile.generate(X)

        normalized = self.normalizer.fit_transform(X)

        return {

            "normalized": normalized,

            "statistics": statistics,

            "quantiles": quantiles

        }


#############################################################
# Example
#############################################################

if __name__ == "__main__":

    np.random.seed(42)

    X = np.random.rand(500, 20) * 500

    model = QSAMMN()

    output = model.process(X)

    print("\nStatistics")

    print(output["statistics"])

    print("\nNormalized Shape")

    print(output["normalized"].shape)

    print("\nFirst Sample")

    print(output["normalized"][0])

"""
===========================================================
SPFI-HC
Part 2B

Diffusion-Based Missing Value Imputation

Equation (2)

Author : SPFI-HC
===========================================================
"""

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


###############################################################
# Observation Mask
###############################################################

class ObservationMask:

    def __init__(self):
        pass

    def create(self, X):

        X = np.asarray(X)

        mask = np.where(

            np.isnan(X),

            0,

            1

        )

        return mask.astype(np.float32)


###############################################################
# Initial Mean Imputation
###############################################################

class InitialImputer:

    def __init__(self):
        pass

    def transform(self, X):

        X = np.asarray(X).copy()

        for col in range(X.shape[1]):

            values = X[:, col]

            mean = np.nanmean(values)

            values[np.isnan(values)] = mean

            X[:, col] = values

        return X


###############################################################
# Affinity Matrix
###############################################################

class AffinityMatrix:

    def __init__(self, sigma=1.0):

        self.sigma = sigma

    def compute(self, X):

        D = euclidean_distances(X)

        A = np.exp(

            -(D ** 2) /

            (2 * self.sigma ** 2)

        )

        return A


###############################################################
# Row Normalization
###############################################################

class RowNormalization:

    def normalize(self, matrix):

        row_sum = matrix.sum(

            axis=1,

            keepdims=True

        )

        row_sum[row_sum == 0] = 1

        return matrix / row_sum


###############################################################
# Diffusion Operator
###############################################################

class DiffusionOperator:

    def __init__(

            self,

            iterations=20,

            alpha=0.85

    ):

        self.iterations = iterations

        self.alpha = alpha

    def diffuse(

            self,

            X,

            mask,

            affinity

    ):

        X = X.copy()

        affinity = RowNormalization().normalize(

            affinity

        )

        original = X.copy()

        for _ in range(self.iterations):

            propagated = affinity @ X

            X = (

                self.alpha * propagated +

                (1 - self.alpha) * original

            )

            X = mask * original + (1-mask) * X

        return X


###############################################################
# Main Diffusion Imputation
###############################################################

class DiffusionImputation:

    """
    Proposed Equation (2)

    X_complete =
    M*X +
    (I-M)*Diffusion(X)

    """

    def __init__(

            self,

            sigma=1.5,

            iterations=25,

            alpha=0.85

    ):

        self.mask_generator = ObservationMask()

        self.initial = InitialImputer()

        self.affinity = AffinityMatrix(

            sigma=sigma

        )

        self.operator = DiffusionOperator(

            iterations=iterations,

            alpha=alpha

        )

    ##########################################################

    def fit_transform(self, X):

        mask = self.mask_generator.create(X)

        X0 = self.initial.transform(X)

        affinity = self.affinity.compute(X0)

        completed = self.operator.diffuse(

            X0,

            mask,

            affinity

        )

        return completed


###############################################################
# Evaluation
###############################################################

class ImputationStatistics:

    def report(

            self,

            original,

            recovered

    ):

        mse = np.mean(

            (original-recovered)**2

        )

        mae = np.mean(

            np.abs(

                original-recovered

            )

        )

        rmse = np.sqrt(mse)

        return {

            "MSE": mse,

            "RMSE": rmse,

            "MAE": mae

        }


###############################################################
# Example
###############################################################

if __name__ == "__main__":

    np.random.seed(10)

    X = np.random.rand(

        300,

        15

    )

    ground_truth = X.copy()

    X[40:80,2] = np.nan
    X[50:100,7] = np.nan
    X[10:90,9] = np.nan

    model = DiffusionImputation(

        sigma=1.2,

        iterations=30,

        alpha=0.90

    )

    recovered = model.fit_transform(X)

    stat = ImputationStatistics()

    result = stat.report(

        ground_truth,

        recovered

    )

    print("\nRecovered Shape")

    print(recovered.shape)

    print("\nStatistics")

    print(result)
"""
=========================================================
SPFI-HC
Part 2C

Edge Preserving Guided Filtering

Equation (3)

Author : SPFI-HC
=========================================================
"""

import numpy as np
from scipy.ndimage import uniform_filter


#############################################################
# Mean Filter
#############################################################

class MeanFilter:

    def __init__(self, radius=5):

        self.radius = radius

    def filter(self, image):

        return uniform_filter(

            image,

            size=self.radius,

            mode="reflect"

        )


#############################################################
# Variance Filter
#############################################################

class VarianceFilter:

    def __init__(self, radius=5):

        self.radius = radius

    def variance(self, image):

        mean = uniform_filter(

            image,

            self.radius

        )

        mean_sq = uniform_filter(

            image * image,

            self.radius

        )

        return mean_sq - mean * mean


#############################################################
# Covariance Filter
#############################################################

class CovarianceFilter:

    def __init__(self, radius=5):

        self.radius = radius

    def covariance(self, I, P):

        mean_I = uniform_filter(I, self.radius)

        mean_P = uniform_filter(P, self.radius)

        mean_IP = uniform_filter(I * P, self.radius)

        return mean_IP - mean_I * mean_P


#############################################################
# Guided Filter
#############################################################

class GuidedFilter:

    """
    Equation (3)

    H = GF(X)

    """

    def __init__(

            self,

            radius=5,

            epsilon=1e-4

    ):

        self.radius = radius

        self.epsilon = epsilon

    #########################################################

    def filter(

            self,

            guidance,

            image

    ):

        guidance = guidance.astype(np.float32)

        image = image.astype(np.float32)

        mean_I = uniform_filter(

            guidance,

            self.radius

        )

        mean_P = uniform_filter(

            image,

            self.radius

        )

        corr_I = uniform_filter(

            guidance * guidance,

            self.radius

        )

        corr_IP = uniform_filter(

            guidance * image,

            self.radius

        )

        var_I = corr_I - mean_I * mean_I

        cov_IP = corr_IP - mean_I * mean_P

        a = cov_IP / (

                var_I +

                self.epsilon

        )

        b = mean_P - a * mean_I

        mean_a = uniform_filter(

            a,

            self.radius

        )

        mean_b = uniform_filter(

            b,

            self.radius

        )

        output = (

                mean_a * guidance +

                mean_b

        )

        return output


#############################################################
# Edge Preservation Score
#############################################################

class EdgePreservationScore:

    def score(

            self,

            original,

            filtered

    ):

        gx1, gy1 = np.gradient(original)

        gx2, gy2 = np.gradient(filtered)

        edge1 = np.sqrt(

            gx1 ** 2 +

            gy1 ** 2

        )

        edge2 = np.sqrt(

            gx2 ** 2 +

            gy2 ** 2

        )

        numerator = np.sum(

            edge1 * edge2

        )

        denominator = np.sqrt(

            np.sum(edge1 ** 2)

            *

            np.sum(edge2 ** 2)

        )

        return numerator / (denominator + 1e-8)


#############################################################
# Noise Reduction Score
#############################################################

class NoiseReduction:

    def mse(

            self,

            original,

            filtered

    ):

        return np.mean(

            (original-filtered)**2

        )

    def psnr(

            self,

            original,

            filtered

    ):

        mse = self.mse(

            original,

            filtered

        )

        if mse == 0:

            return 100

        return 20*np.log10(1.0/np.sqrt(mse))


#############################################################
# Complete Guided Filter Pipeline
#############################################################

class EdgePreservingGuidedFilter:

    def __init__(

            self,

            radius=5,

            epsilon=1e-4

    ):

        self.guided = GuidedFilter(

            radius,

            epsilon

        )

        self.edge = EdgePreservationScore()

        self.noise = NoiseReduction()

    #########################################################

    def process(

            self,

            image

    ):

        filtered = self.guided.filter(

            image,

            image

        )

        score = self.edge.score(

            image,

            filtered

        )

        mse = self.noise.mse(

            image,

            filtered

        )

        psnr = self.noise.psnr(

            image,

            filtered

        )

        return {

            "filtered": filtered,

            "edge_score": score,

            "mse": mse,

            "psnr": psnr

        }


#############################################################
# Example
#############################################################

if __name__ == "__main__":

    np.random.seed(1)

    img = np.random.rand(

        256,

        256

    )

    noise = np.random.normal(

        0,

        0.05,

        img.shape

    )

    noisy = img + noise

    model = EdgePreservingGuidedFilter(

        radius=7,

        epsilon=1e-4

    )

    result = model.process(noisy)

    print("Filtered Shape :", result["filtered"].shape)
    print("Edge Score :", result["edge_score"])
    print("MSE :", result["mse"])
    print("PSNR :", result["psnr"])

"""
=============================================================
SPFI-HC
Part 3A

Swin Transformer V2 Feature Extractor

Equation (4)

Fi = SwinTransformer(Ximage)

=============================================================
"""

import torch
import torch.nn as nn
import timm


###############################################################
# Swin Transformer V2 Encoder
###############################################################

class SwinV2Encoder(nn.Module):

    def __init__(

            self,

            model_name="swinv2_tiny_window8_256",

            pretrained=True,

            embedding_dim=768,

            dropout=0.30

    ):

        super().__init__()

        #######################################################

        self.backbone = timm.create_model(

            model_name,

            pretrained=pretrained,

            num_classes=0,

            global_pool="avg"

        )

        #######################################################

        in_features = self.backbone.num_features

        #######################################################

        self.embedding = nn.Sequential(

            nn.Linear(

                in_features,

                embedding_dim

            ),

            nn.LayerNorm(

                embedding_dim

            ),

            nn.GELU(),

            nn.Dropout(

                dropout

            )

        )

    ###########################################################

    def forward(self, x):

        feature = self.backbone(x)

        embedding = self.embedding(feature)

        return embedding


###############################################################
# Image Projection Head
###############################################################

class ImageProjection(nn.Module):

    def __init__(

            self,

            input_dim=768,

            projection_dim=512

    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(

                input_dim,

                1024

            ),

            nn.GELU(),

            nn.Linear(

                1024,

                projection_dim

            )

        )

    def forward(self, x):

        return self.net(x)


###############################################################
# Image Feature Normalizer
###############################################################

class FeatureNormalizer(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, x):

        return nn.functional.normalize(

            x,

            dim=-1

        )


###############################################################
# Complete Image Encoder
###############################################################

class MedicalImageEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = SwinV2Encoder()

        self.project = ImageProjection()

        self.normalize = FeatureNormalizer()

    def forward(self, image):

        feature = self.encoder(image)

        feature = self.project(feature)

        feature = self.normalize(feature)

        return feature


###############################################################
# Example
###############################################################

if __name__ == "__main__":

    model = MedicalImageEncoder()

    x = torch.randn(

        4,

        3,

        256,

        256

    )

    output = model(x)

    print()

    print("Output Shape")

    print(output.shape)

"""
=========================================================
SPFI-HC
Part 3B

Mamba State Space Network
Physiological / IoT Signal Encoder

Equation (5)

Fs = Mamba(Xsignal)

=========================================================
"""

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


############################################################
# Signal Embedding
############################################################

class SignalEmbedding(nn.Module):

    def __init__(self,
                 input_dim=13,
                 embed_dim=256):

        super().__init__()

        self.embedding = nn.Sequential(

            nn.Linear(input_dim,128),

            nn.GELU(),

            nn.Linear(128,embed_dim),

            nn.LayerNorm(embed_dim)

        )

    def forward(self,x):

        return self.embedding(x)


############################################################
# Mamba Block
############################################################

class MambaBlock(nn.Module):

    def __init__(self,
                 dim=256):

        super().__init__()

        if Mamba is not None:

            self.block = Mamba(

                d_model=dim,

                d_state=16,

                d_conv=4,

                expand=2

            )

        else:

            # fallback if mamba_ssm isn't installed
            self.block = nn.GRU(

                dim,

                dim,

                batch_first=True

            )

    def forward(self,x):

        if Mamba is not None:

            return self.block(x)

        else:

            y,_ = self.block(x)

            return y


############################################################
# Feed Forward Network
############################################################

class FeedForward(nn.Module):

    def __init__(self,
                 dim=256):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(dim,dim*4),

            nn.GELU(),

            nn.Dropout(0.2),

            nn.Linear(dim*4,dim)

        )

    def forward(self,x):

        return self.net(x)


############################################################
# Residual Block
############################################################

class ResidualMamba(nn.Module):

    def __init__(self,
                 dim=256):

        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.mamba = MambaBlock(dim)

        self.norm2 = nn.LayerNorm(dim)

        self.ffn = FeedForward(dim)

    def forward(self,x):

        y = self.mamba(self.norm1(x))

        x = x + y

        y = self.ffn(self.norm2(x))

        x = x + y

        return x


############################################################
# Temporal Pooling
############################################################

class TemporalPooling(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self,x):

        return torch.mean(

            x,

            dim=1

        )


############################################################
# Projection Head
############################################################

class ProjectionHead(nn.Module):

    def __init__(self,
                 dim=256,
                 output_dim=512):

        super().__init__()

        self.head = nn.Sequential(

            nn.Linear(dim,512),

            nn.GELU(),

            nn.Dropout(0.3),

            nn.Linear(512,output_dim)

        )

    def forward(self,x):

        return self.head(x)


############################################################
# Complete Mamba Encoder
############################################################

class MambaEncoder(nn.Module):

    def __init__(self,
                 input_dim=13):

        super().__init__()

        self.embedding = SignalEmbedding(

            input_dim=input_dim

        )

        self.encoder = nn.Sequential(

            ResidualMamba(),

            ResidualMamba(),

            ResidualMamba(),

            ResidualMamba()

        )

        self.pool = TemporalPooling()

        self.project = ProjectionHead()

    def forward(self,x):

        """
        Input

        Batch × Time × Features

        """

        x = self.embedding(x)

        x = self.encoder(x)

        x = self.pool(x)

        x = self.project(x)

        x = nn.functional.normalize(

            x,

            dim=-1

        )

        return x


############################################################
# Example
############################################################

if __name__ == "__main__":

    model = MambaEncoder(

        input_dim=13

    )

    signal = torch.randn(

        8,

        100,

        13

    )

    feature = model(signal)

    print()

    print("Feature Shape")

    print(feature.shape)

"""
=========================================================
SPFI-HC
Part 3C

BioMedLM Clinical Text Encoder

Equation (6)

Ft = BioMedLM(Xtext)

=========================================================
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


############################################################
# BioMedLM Encoder
############################################################

class BioMedLMEncoder(nn.Module):

    def __init__(
            self,
            model_name="stanford-crfm/BioMedLM",
            embedding_dim=768,
            projection_dim=512,
            max_length=256,
            dropout=0.3
    ):

        super().__init__()

        self.max_length = max_length

        ####################################################

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModel.from_pretrained(
            model_name
        )

        ####################################################

        hidden_size = self.model.config.hidden_size

        self.projection = nn.Sequential(

            nn.Linear(hidden_size, embedding_dim),

            nn.LayerNorm(embedding_dim),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(embedding_dim, projection_dim)

        )

    ####################################################

    def tokenize(self, text):

        encoded = self.tokenizer(

            text,

            padding="max_length",

            truncation=True,

            max_length=self.max_length,

            return_tensors="pt"

        )

        return encoded

    ####################################################

    def forward(self, text):

        device = next(self.parameters()).device

        token = self.tokenize(text)

        token = {

            k: v.to(device)

            for k, v in token.items()

        }

        outputs = self.model(**token)

        cls = outputs.last_hidden_state[:,0,:]

        feature = self.projection(cls)

        feature = nn.functional.normalize(

            feature,

            dim=-1

        )

        return feature


############################################################
# Clinical Text Encoder Wrapper
############################################################

class ClinicalTextEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = BioMedLMEncoder()

    def forward(self,text):

        return self.encoder(text)


############################################################
# Batch Encoder
############################################################

class BatchClinicalEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = ClinicalTextEncoder()

    def forward(self,texts):

        outputs=[]

        for t in texts:

            feature=self.encoder(t)

            outputs.append(feature)

        outputs=torch.cat(outputs,dim=0)

        return outputs


############################################################
# Example
############################################################

if __name__=="__main__":

    model=ClinicalTextEncoder()

    sample="Patient shows symptoms of pneumonia with elevated temperature."

    with torch.no_grad():

        feature=model(sample)

    print()

    print("Feature Shape")

    print(feature.shape)

"""
=========================================================
SPFI-HC
Part 3D

FT-Transformer
Electronic Health Record Encoder

Equation (7)

Fe = FTTransformer(Xehr)

=========================================================
"""

import torch
import torch.nn as nn


############################################################
# Numerical Feature Embedding
############################################################

class NumericalEmbedding(nn.Module):

    def __init__(
            self,
            input_dim=64,
            embed_dim=256):

        super().__init__()

        self.embedding = nn.Sequential(

            nn.Linear(input_dim,512),

            nn.GELU(),

            nn.Linear(512,embed_dim),

            nn.LayerNorm(embed_dim)

        )

    def forward(self,x):

        return self.embedding(x)


############################################################
# Multi Head Attention
############################################################

class MultiHeadAttention(nn.Module):

    def __init__(
            self,
            embed_dim=256,
            heads=8):

        super().__init__()

        self.attention = nn.MultiheadAttention(

            embed_dim,

            heads,

            batch_first=True

        )

    def forward(self,x):

        out,_ = self.attention(

            x,

            x,

            x

        )

        return out


############################################################
# Feed Forward Network
############################################################

class FeedForward(nn.Module):

    def __init__(
            self,
            embed_dim=256):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(embed_dim,embed_dim*4),

            nn.GELU(),

            nn.Dropout(0.2),

            nn.Linear(embed_dim*4,embed_dim)

        )

    def forward(self,x):

        return self.network(x)


############################################################
# Transformer Block
############################################################

class TransformerBlock(nn.Module):

    def __init__(
            self,
            embed_dim=256):

        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)

        self.attn = MultiHeadAttention(embed_dim)

        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = FeedForward(embed_dim)

    def forward(self,x):

        x = x + self.attn(

            self.norm1(x)

        )

        x = x + self.ffn(

            self.norm2(x)

        )

        return x


############################################################
# FT Transformer
############################################################

class FTTransformer(nn.Module):

    def __init__(
            self,
            input_dim=64):

        super().__init__()

        self.embedding = NumericalEmbedding(

            input_dim

        )

        self.encoder = nn.Sequential(

            TransformerBlock(),

            TransformerBlock(),

            TransformerBlock(),

            TransformerBlock()

        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.project = nn.Sequential(

            nn.Linear(256,512),

            nn.GELU(),

            nn.Dropout(0.3),

            nn.Linear(512,512)

        )

    ########################################################

    def forward(self,x):

        x = self.embedding(x)

        x = x.unsqueeze(1)

        x = self.encoder(x)

        x = x.transpose(1,2)

        x = self.pool(x)

        x = x.squeeze(-1)

        x = self.project(x)

        x = nn.functional.normalize(

            x,

            dim=-1

        )

        return x


############################################################
# Complete EHR Encoder
############################################################

class EHREncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = FTTransformer()

    def forward(self,x):

        return self.encoder(x)


############################################################
# Example
############################################################

if __name__=="__main__":

    model = EHREncoder()

    ehr = torch.randn(

        8,

        64

    )

    feature = model(ehr)

    print()

    print("Feature Shape")

    print(feature.shape)
"""
=============================================================
SPFI-HC

Part 3E

Multimodal Representation Learning

Equation (8)

F = Concat(Fimage,Fsignal,Ftext,Fehr)

=============================================================
"""

import torch
import torch.nn as nn


############################################################
# Cross Modal Self Attention
############################################################

class CrossModalAttention(nn.Module):

    def __init__(self,
                 embed_dim=512,
                 heads=8):

        super().__init__()

        self.attention = nn.MultiheadAttention(

            embed_dim,

            heads,

            batch_first=True

        )

    def forward(self,x):

        y,_ = self.attention(

            x,

            x,

            x

        )

        return y


############################################################
# Feed Forward
############################################################

class FeedForward(nn.Module):

    def __init__(self,
                 dim=512):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(dim,2048),

            nn.GELU(),

            nn.Dropout(0.30),

            nn.Linear(2048,dim)

        )

    def forward(self,x):

        return self.net(x)


############################################################
# Transformer Fusion Block
############################################################

class FusionBlock(nn.Module):

    def __init__(self):

        super().__init__()

        self.norm1 = nn.LayerNorm(512)

        self.attention = CrossModalAttention()

        self.norm2 = nn.LayerNorm(512)

        self.ffn = FeedForward()

    def forward(self,x):

        x = x + self.attention(

            self.norm1(x)

        )

        x = x + self.ffn(

            self.norm2(x)

        )

        return x


############################################################
# Adaptive Feature Weighting
############################################################

class AdaptiveWeighting(nn.Module):

    def __init__(self):

        super().__init__()

        self.weight = nn.Sequential(

            nn.Linear(512,128),

            nn.ReLU(),

            nn.Linear(128,1),

            nn.Sigmoid()

        )

    def forward(self,x):

        w = self.weight(x)

        return x*w


############################################################
# Global Pooling
############################################################

class GlobalPooling(nn.Module):

    def forward(self,x):

        return torch.mean(

            x,

            dim=1

        )


############################################################
# Projection
############################################################

class ProjectionHead(nn.Module):

    def __init__(self):

        super().__init__()

        self.project = nn.Sequential(

            nn.Linear(

                512,

                1024

            ),

            nn.GELU(),

            nn.Dropout(

                0.30

            ),

            nn.Linear(

                1024,

                512

            )

        )

    def forward(self,x):

        return self.project(x)


############################################################
# Complete Multimodal Fusion
############################################################

class MultimodalFusion(nn.Module):

    def __init__(self):

        super().__init__()

        self.weight = AdaptiveWeighting()

        self.block1 = FusionBlock()

        self.block2 = FusionBlock()

        self.block3 = FusionBlock()

        self.pool = GlobalPooling()

        self.project = ProjectionHead()

    ########################################################

    def forward(

            self,

            image_feature,

            signal_feature,

            text_feature,

            ehr_feature

    ):

        x = torch.stack([

            image_feature,

            signal_feature,

            text_feature,

            ehr_feature

        ],dim=1)

        x = self.weight(x)

        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        x = self.pool(x)

        x = self.project(x)

        x = nn.functional.normalize(

            x,

            dim=-1

        )

        return x


############################################################
# Example
############################################################

if __name__=="__main__":

    image = torch.randn(8,512)

    signal = torch.randn(8,512)

    text = torch.randn(8,512)

    ehr = torch.randn(8,512)

    model = MultimodalFusion()

    output = model(

        image,

        signal,

        text,

        ehr

    )

    print()

    print("Unified Feature")

    print(output.shape)

"""
==============================================================
SPFI-HC

Part 4A

Modified Sensitivity Impact Support Vector Analysis (MSI-SVA)

Equation (9)

Privacy Sensitivity Projection

==============================================================
"""

import numpy as np
import torch
import torch.nn as nn


############################################################
# Projection Network
############################################################

class PrivacyProjection(nn.Module):

    def __init__(self,
                 input_dim=512,
                 hidden_dim=256):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_dim, hidden_dim),

            nn.GELU(),

            nn.Dropout(0.30),

            nn.Linear(hidden_dim, hidden_dim),

            nn.GELU(),

            nn.Linear(hidden_dim, 1)

        )

    def forward(self, x):

        return self.network(x)


############################################################
# L2 Normalization
############################################################

class SensitivityNormalization(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, score):

        norm = torch.norm(

            score,

            p=2,

            dim=0,

            keepdim=True

        ) + 1e-8

        return score / norm


############################################################
# Privacy Score Generator
############################################################

class PrivacyScoreGenerator(nn.Module):

    def __init__(self):

        super().__init__()

        self.projector = PrivacyProjection()

        self.normalize = SensitivityNormalization()

    def forward(self, feature):

        score = self.projector(feature)

        score = self.normalize(score)

        return score


############################################################
# Privacy Risk Analyzer
############################################################

class PrivacyRiskAnalyzer:

    def __init__(self):

        pass

    def analyze(self, score):

        score = score.detach().cpu().numpy()

        report = {

            "Minimum": float(np.min(score)),

            "Maximum": float(np.max(score)),

            "Average": float(np.mean(score)),

            "Std": float(np.std(score))

        }

        return report


############################################################
# Feature Ranking
############################################################

class FeatureRanking:

    def rank(self, score):

        score = score.squeeze()

        index = torch.argsort(

            score,

            descending=True

        )

        return index


############################################################
# MSI-SVA Module
############################################################

class MSISVA(nn.Module):

    def __init__(self):

        super().__init__()

        self.generator = PrivacyScoreGenerator()

        self.ranker = FeatureRanking()

        self.analysis = PrivacyRiskAnalyzer()

    def forward(self, feature):

        score = self.generator(feature)

        ranking = self.ranker.rank(score)

        report = self.analysis.analyze(score)

        return {

            "privacy_score": score,

            "ranking": ranking,

            "statistics": report

        }


############################################################
# Example
############################################################

if __name__ == "__main__":

    feature = torch.randn(

        64,

        512

    )

    model = MSISVA()

    output = model(feature)

    print()

    print("Privacy Score Shape")

    print(output["privacy_score"].shape)

    print()

    print("Top 10 Sensitive Features")

    print(output["ranking"][:10])

    print()

    print(output["statistics"])

"""
==============================================================
SPFI-HC

Part 4B

Sensitivity-Aware Feature Representation

Equation (10)

F_sensitive = F × S

==============================================================
"""

import torch
import torch.nn as nn
import numpy as np


############################################################
# Adaptive Privacy Threshold
############################################################

class AdaptiveThreshold(nn.Module):

    def __init__(self, percentile=70):

        super().__init__()

        self.percentile = percentile

    def forward(self, scores):

        threshold = torch.quantile(

            scores,

            self.percentile / 100.0

        )

        return threshold


############################################################
# Privacy Mask Generator
############################################################

class PrivacyMask(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(

            self,

            scores,

            threshold

    ):

        mask = (

            scores >= threshold

        ).float()

        return mask


############################################################
# Feature Weighting
############################################################

class FeatureWeighting(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(

            self,

            feature,

            score

    ):

        return feature * score


############################################################
# Sensitivity Representation
############################################################

class SensitivityRepresentation(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(

            self,

            feature,

            mask

    ):

        return feature * mask


############################################################
# Privacy Statistics
############################################################

class PrivacyStatistics:

    def report(

            self,

            scores

    ):

        score = scores.detach().cpu().numpy()

        return {

            "minimum": float(score.min()),

            "maximum": float(score.max()),

            "mean": float(score.mean()),

            "std": float(score.std()),

            "median": float(np.median(score))

        }


############################################################
# MSI-SVA Stage-2
############################################################

class SensitivityAwareFeature(nn.Module):

    def __init__(

            self,

            percentile=70

    ):

        super().__init__()

        self.threshold = AdaptiveThreshold(

            percentile

        )

        self.mask = PrivacyMask()

        self.weight = FeatureWeighting()

        self.feature = SensitivityRepresentation()

        self.statistics = PrivacyStatistics()

    ########################################################

    def forward(

            self,

            feature,

            score

    ):

        threshold = self.threshold(score)

        mask = self.mask(

            score,

            threshold

        )

        weighted = self.weight(

            feature,

            score

        )

        output = self.feature(

            weighted,

            mask

        )

        report = self.statistics.report(

            score

        )

        return {

            "feature": output,

            "mask": mask,

            "threshold": threshold,

            "statistics": report

        }


############################################################
# Example
############################################################

if __name__ == "__main__":

    feature = torch.randn(

        64,

        512

    )

    score = torch.rand(

        64,

        1

    )

    model = SensitivityAwareFeature()

    output = model(

        feature,

        score

    )

    print()

    print("Sensitivity Feature Shape")

    print(output["feature"].shape)

    print()

    print("Mask Shape")

    print(output["mask"].shape)

    print()

    print("Threshold")

    print(output["threshold"])

    print()

    print(output["statistics"])
"""
==============================================================

SPFI-HC

Part 5A

Modified Swarm Intelligence Feature Selection Algorithm

Equation (11)

Feature Fitness Evaluation

==============================================================
"""

import torch
import torch.nn as nn
import numpy as np


############################################################
# Fitness Function
############################################################

class FeatureFitness(nn.Module):

    """
    Equation (11)

    Computes feature fitness score

    """

    def __init__(self):

        super().__init__()

    def forward(

            self,

            feature,

            privacy_score

    ):

        importance = torch.mean(

            torch.abs(feature),

            dim=0

        )

        privacy = torch.mean(

            privacy_score

        )

        fitness = importance * (1.0 - privacy)

        return fitness


############################################################
# Swarm Particle
############################################################

class SwarmParticle:

    def __init__(

            self,

            dimension

    ):

        self.position = torch.rand(

            dimension

        )

        self.velocity = torch.randn(

            dimension

        ) * 0.05

        self.best_position = self.position.clone()

        self.best_score = -1e9


############################################################
# Swarm Initialization
############################################################

class SwarmInitialization:

    def __init__(

            self,

            particles=30

    ):

        self.particles = particles

    def initialize(

            self,

            dimension

    ):

        swarm = []

        for _ in range(self.particles):

            swarm.append(

                SwarmParticle(dimension)

            )

        return swarm


############################################################
# Fitness Evaluation
############################################################

class SwarmEvaluator:

    def evaluate(

            self,

            particle,

            fitness

    ):

        score = torch.sum(

            particle.position * fitness

        )

        return score.item()


############################################################
# Global Best
############################################################

class GlobalBest:

    def __init__(self):

        self.position = None

        self.score = -1e9

    def update(

            self,

            particle,

            score

    ):

        if score > self.score:

            self.score = score

            self.position = particle.position.clone()


############################################################
# Velocity Update
############################################################

class VelocityUpdate:

    def __init__(

            self,

            w=0.7,

            c1=1.5,

            c2=1.5

    ):

        self.w = w

        self.c1 = c1

        self.c2 = c2

    def update(

            self,

            particle,

            global_best

    ):

        r1 = torch.rand_like(

            particle.velocity

        )

        r2 = torch.rand_like(

            particle.velocity

        )

        cognitive = (

            self.c1 *

            r1 *

            (

                particle.best_position -

                particle.position

            )

        )

        social = (

            self.c2 *

            r2 *

            (

                global_best -

                particle.position

            )

        )

        particle.velocity = (

            self.w *

            particle.velocity +

            cognitive +

            social

        )

        particle.position += particle.velocity

        particle.position = torch.clamp(

            particle.position,

            0,

            1

        )


############################################################
# MSI-FSA
############################################################

class MSIFSA(nn.Module):

    def __init__(

            self,

            particles=30,

            iterations=40

    ):

        super().__init__()

        self.iterations = iterations

        self.fitness = FeatureFitness()

        self.init = SwarmInitialization(

            particles

        )

        self.eval = SwarmEvaluator()

        self.velocity = VelocityUpdate()

    ########################################################

    def forward(

            self,

            feature,

            privacy_score

    ):

        fitness = self.fitness(

            feature,

            privacy_score

        )

        dimension = fitness.shape[0]

        swarm = self.init.initialize(

            dimension

        )

        global_best = GlobalBest()

        ####################################################

        for _ in range(self.iterations):

            for particle in swarm:

                score = self.eval.evaluate(

                    particle,

                    fitness

                )

                if score > particle.best_score:

                    particle.best_score = score

                    particle.best_position = particle.position.clone()

                global_best.update(

                    particle,

                    score

                )

            for particle in swarm:

                self.velocity.update(

                    particle,

                    global_best.position

                )

        return {

            "fitness": fitness,

            "best_position": global_best.position,

            "best_score": global_best.score

        }


############################################################
# Example
############################################################

if __name__ == "__main__":

    feature = torch.randn(

        128,

        512

    )

    privacy = torch.rand(

        128,

        1

    )

    model = MSIFSA(

        particles=20,

        iterations=30

    )

    result = model(

        feature,

        privacy

    )

    print()

    print("Fitness Shape")

    print(result["fitness"].shape)

    print()

    print("Global Best Score")

    print(result["best_score"])

    print()

    print("Selected Feature Vector")

    print(result["best_position"].shape)
"""
==============================================================
SPFI-HC

Part 5B

Modified Swarm Intelligence Feature Selection

Equation (12)

Binary Feature Selection

==============================================================
"""

import torch
import torch.nn as nn


############################################################
# Sigmoid Transfer Function
############################################################

class SigmoidTransfer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, position):
        return torch.sigmoid(position)


############################################################
# Binary Threshold
############################################################

class BinaryThreshold(nn.Module):

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, probability):

        binary = (probability >= self.threshold).float()

        return binary


############################################################
# Feature Selector
############################################################

class FeatureSelector(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, feature, mask):

        return feature * mask.unsqueeze(0)


############################################################
# Selected Feature Index
############################################################

class SelectedFeatureIndex:

    def get_index(self, mask):

        return torch.nonzero(mask).flatten()


############################################################
# Feature Statistics
############################################################

class FeatureStatistics:

    def report(self, mask):

        selected = int(mask.sum().item())

        total = len(mask)

        removed = total - selected

        ratio = selected / total

        return {

            "Total Features": total,

            "Selected Features": selected,

            "Removed Features": removed,

            "Selection Ratio": ratio

        }


############################################################
# Binary Feature Selection Module
############################################################

class BinaryFeatureSelection(nn.Module):

    def __init__(self, threshold=0.5):

        super().__init__()

        self.sigmoid = SigmoidTransfer()

        self.threshold = BinaryThreshold(threshold)

        self.selector = FeatureSelector()

        self.index = SelectedFeatureIndex()

        self.statistics = FeatureStatistics()

    ########################################################

    def forward(

            self,

            feature,

            best_position

    ):

        probability = self.sigmoid(best_position)

        mask = self.threshold(probability)

        selected_feature = self.selector(

            feature,

            mask

        )

        selected_index = self.index.get_index(mask)

        report = self.statistics.report(mask)

        return {

            "selected_feature": selected_feature,

            "mask": mask,

            "index": selected_index,

            "statistics": report

        }


############################################################
# Example
############################################################

if __name__ == "__main__":

    feature = torch.randn(

        64,

        512

    )

    best_position = torch.randn(

        512

    )

    model = BinaryFeatureSelection(

        threshold=0.5

    )

    result = model(

        feature,

        best_position

    )

    print()

    print("Selected Feature Shape")

    print(result["selected_feature"].shape)

    print()

    print("Selected Indices")

    print(result["index"][:20])

    print()

    print(result["statistics"])

"""
==============================================================
SPFI-HC

Part 5C

Modified Swarm Intelligence Feature Selection (MSI-FSA)

Equation (13)

Final Feature Refinement

Author : SPFI-HC

==============================================================
"""

import torch
import torch.nn as nn


############################################################
# Feature Importance
############################################################

class FeatureImportance(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, feature):

        importance = torch.mean(

            torch.abs(feature),

            dim=0

        )

        return importance


############################################################
# Redundancy Removal
############################################################

class RedundancyRemoval(nn.Module):

    def __init__(self,
                 threshold=0.95):

        super().__init__()

        self.threshold = threshold

    def forward(self, feature):

        corr = torch.corrcoef(

            feature.T

        )

        keep = torch.ones(

            corr.shape[0],

            dtype=torch.bool,

            device=feature.device

        )

        for i in range(corr.shape[0]):

            if not keep[i]:
                continue

            for j in range(i + 1, corr.shape[0]):

                if torch.abs(corr[i, j]) >= self.threshold:
                    keep[j] = False

        return feature[:, keep], keep


############################################################
# Feature Ranking
############################################################

class FeatureRanking(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, importance):

        ranking = torch.argsort(

            importance,

            descending=True

        )

        return ranking


############################################################
# Top-K Selection
############################################################

class TopKSelector(nn.Module):

    def __init__(self,
                 top_k=256):

        super().__init__()

        self.top_k = top_k

    def forward(self,
                feature,
                importance):

        k = min(

            self.top_k,

            feature.shape[1]

        )

        index = torch.argsort(

            importance,

            descending=True

        )[:k]

        refined = feature[:, index]

        return refined, index


############################################################
# Statistics
############################################################

class RefinementStatistics:

    def report(

            self,

            original,

            refined

    ):

        return {

            "Original Features": original,

            "Final Features": refined,

            "Compression Ratio":

                refined / original

        }


############################################################
# Complete Refinement Module
############################################################

class FeatureRefinement(nn.Module):

    """
    Equation (13)

    Final Optimized Feature Set
    """

    def __init__(

            self,

            redundancy_threshold=0.95,

            top_k=256

    ):

        super().__init__()

        self.importance = FeatureImportance()

        self.redundancy = RedundancyRemoval(

            redundancy_threshold

        )

        self.rank = FeatureRanking()

        self.selector = TopKSelector(

            top_k

        )

        self.statistics = RefinementStatistics()

    ########################################################

    def forward(self, feature):

        importance = self.importance(

            feature

        )

        feature, keep = self.redundancy(

            feature

        )

        importance = importance[keep]

        ranking = self.rank(

            importance

        )

        refined, selected = self.selector(

            feature,

            importance

        )

        report = self.statistics.report(

            feature.shape[1],

            refined.shape[1]

        )

        return {

            "feature": refined,

            "importance": importance,

            "ranking": ranking,

            "selected_index": selected,

            "statistics": report

        }


############################################################
# Example
############################################################

if __name__ == "__main__":

    torch.manual_seed(42)

    feature = torch.randn(

        128,

        512

    )

    model = FeatureRefinement(

        redundancy_threshold=0.90,

        top_k=256

    )

    result = model(

        feature

    )

    print()

    print("Final Feature Shape")

    print(result["feature"].shape)

    print()

    print("Top 20 Features")

    print(result["selected_index"][:20])

    print()

    print(result["statistics"])

"""
==============================================================
SPFI-HC

Part 6A

Blockchain Block Structure

==============================================================
"""

import hashlib
import json
import time


class Block:

    def __init__(
            self,
            index,
            transactions,
            timestamp,
            previous_hash,
            nonce=0
    ):

        self.index = index
        self.transactions = transactions
        self.timestamp = timestamp
        self.previous_hash = previous_hash
        self.nonce = nonce

        self.hash = self.compute_hash()

    ##########################################################

    def compute_hash(self):

        block = {

            "index": self.index,
            "transactions": self.transactions,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce

        }

        encoded = json.dumps(
            block,
            sort_keys=True
        ).encode()

        return hashlib.sha256(
            encoded
        ).hexdigest()


##############################################################
# Blockchain
##############################################################

class Blockchain:

    difficulty = 4

    def __init__(self):

        self.chain = []

        self.pending_transactions = []

        self.create_genesis_block()

    ##########################################################

    def create_genesis_block(self):

        genesis = Block(

            index=0,

            transactions=[],

            timestamp=time.time(),

            previous_hash="0"

        )

        self.chain.append(genesis)

    ##########################################################

    @property
    def last_block(self):

        return self.chain[-1]

    ##########################################################

    def add_transaction(self, transaction):

        self.pending_transactions.append(transaction)

    ##########################################################

    def proof_of_work(self, block):

        block.nonce = 0

        hash_value = block.compute_hash()

        while not hash_value.startswith(

                "0" * self.difficulty

        ):

            block.nonce += 1

            hash_value = block.compute_hash()

        return hash_value

    ##########################################################

    def add_block(self):

        block = Block(

            index=len(self.chain),

            transactions=self.pending_transactions,

            timestamp=time.time(),

            previous_hash=self.last_block.hash

        )

        proof = self.proof_of_work(block)

        block.hash = proof

        self.chain.append(block)

        self.pending_transactions = []

    ##########################################################

    def validate(self):

        for i in range(

                1,

                len(self.chain)

        ):

            current = self.chain[i]

            previous = self.chain[i - 1]

            if current.previous_hash != previous.hash:

                return False

            if current.hash != current.compute_hash():

                return False

        return True

    ##########################################################

    def display(self):

        for block in self.chain:

            print("=" * 60)

            print("Index :", block.index)

            print("Hash :", block.hash)

            print("Previous :", block.previous_hash)

            print("Transactions :", block.transactions)

            print()


##############################################################
# Example
##############################################################

if __name__ == "__main__":

    blockchain = Blockchain()

    blockchain.add_transaction({

        "Hospital": "Hospital_A",

        "Round": 1,

        "Accuracy": 98.95

    })

    blockchain.add_transaction({

        "Hospital": "Hospital_B",

        "Round": 1,

        "Accuracy": 99.10

    })

    blockchain.add_block()

    blockchain.display()

    print()

    print("Blockchain Valid :", blockchain.validate())

"""
==============================================================
SPFI-HC

Part 6B

Blockchain Enabled Federated Learning

FedAvg Aggregation

==============================================================
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim


##############################################################
# Local Client
##############################################################

class FederatedClient:

    def __init__(
            self,
            client_id,
            model,
            train_loader,
            device="cuda"
    ):

        self.client_id = client_id
        self.device = device

        self.model = copy.deepcopy(model).to(device)

        self.loader = train_loader

    ##########################################################

    def train_local(

            self,

            epochs=1,

            lr=1e-4

    ):

        optimizer = optim.Adam(

            self.model.parameters(),

            lr=lr

        )

        criterion = nn.CrossEntropyLoss()

        self.model.train()

        epoch_loss = 0

        for _ in range(epochs):

            for batch in self.loader:

                image = batch["image"].to(self.device)
                label = batch["label"].to(self.device)

                optimizer.zero_grad()

                output = self.model(image)

                loss = criterion(

                    output,

                    label

                )

                loss.backward()

                optimizer.step()

                epoch_loss += loss.item()

        return epoch_loss

    ##########################################################

    def get_weights(self):

        return copy.deepcopy(

            self.model.state_dict()

        )

    ##########################################################

    def set_weights(

            self,

            weights

    ):

        self.model.load_state_dict(

            weights

        )


##############################################################
# Federated Averaging
##############################################################

class FedAvg:

    def __init__(self):
        pass

    def aggregate(

            self,

            local_models

    ):

        global_weights = copy.deepcopy(

            local_models[0]

        )

        for key in global_weights.keys():

            for i in range(

                    1,

                    len(local_models)

            ):

                global_weights[key] += local_models[i][key]

            global_weights[key] /= len(local_models)

        return global_weights


##############################################################
# Server
##############################################################

class FederatedServer:

    def __init__(

            self,

            global_model,

            blockchain

    ):

        self.global_model = global_model

        self.blockchain = blockchain

        self.aggregator = FedAvg()

    ##########################################################

    def communication_round(

            self,

            clients,

            round_number

    ):

        local_weights = []

        ######################################################

        for client in clients:

            loss = client.train_local()

            weights = client.get_weights()

            local_weights.append(

                weights

            )

            self.blockchain.add_transaction({

                "Client": client.client_id,

                "Round": round_number,

                "Loss": float(loss)

            })

        ######################################################

        global_weights = self.aggregator.aggregate(

            local_weights

        )

        self.global_model.load_state_dict(

            global_weights

        )

        ######################################################

        for client in clients:

            client.set_weights(

                global_weights

            )

        ######################################################

        self.blockchain.add_block()

        return global_weights


##############################################################
# Trainer
##############################################################

class FederatedTrainer:

    def __init__(

            self,

            server,

            clients

    ):

        self.server = server

        self.clients = clients

    ##########################################################

    def train(

            self,

            rounds=10

    ):

        for r in range(rounds):

            print()

            print("="*60)

            print("Federated Round", r+1)

            print("="*60)

            self.server.communication_round(

                self.clients,

                r+1

            )

        print()

        print("Training Finished")


##############################################################
# Example
##############################################################

if __name__ == "__main__":

    print()

    print("Federated Learning Module Ready")

"""
==============================================================
SPFI-HC

Part 6C

Blockchain Security
Digital Signature & Model Authentication

==============================================================
"""

import hashlib
import pickle
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.exceptions import InvalidSignature


##############################################################
# Key Generator
##############################################################

class KeyManager:

    def __init__(self):

        self.private_key = ec.generate_private_key(

            ec.SECP256R1()

        )

        self.public_key = self.private_key.public_key()

    ##########################################################

    def get_private_key(self):

        return self.private_key

    ##########################################################

    def get_public_key(self):

        return self.public_key


##############################################################
# Model Hash Generator
##############################################################

class ModelHasher:

    @staticmethod
    def hash_model(model_state):

        data = pickle.dumps(model_state)

        return hashlib.sha256(data).hexdigest()


##############################################################
# Digital Signature
##############################################################

class DigitalSignature:

    def __init__(self, private_key):

        self.private_key = private_key

    ##########################################################

    def sign(self, message):

        if isinstance(message, str):
            message = message.encode()

        signature = self.private_key.sign(

            message,

            ec.ECDSA(hashes.SHA256())

        )

        return signature


##############################################################
# Signature Verification
##############################################################

class SignatureVerifier:

    def __init__(self, public_key):

        self.public_key = public_key

    ##########################################################

    def verify(

            self,

            message,

            signature

    ):

        if isinstance(message, str):
            message = message.encode()

        try:

            self.public_key.verify(

                signature,

                message,

                ec.ECDSA(hashes.SHA256())

            )

            return True

        except InvalidSignature:

            return False


##############################################################
# Secure Transaction
##############################################################

class SecureTransaction:

    def __init__(

            self,

            sender,

            round_id,

            model_hash,

            signature

    ):

        self.sender = sender

        self.round_id = round_id
        self.model_hash = model_hash
        self.signature = signature

    ##########################################################

    def to_dict(self):

        return {

            "sender": self.sender,

            "round": self.round_id,

            "model_hash": self.model_hash,

            "signature":

                self.signature.hex()

        }


##############################################################
# Authentication Manager
##############################################################

class AuthenticationManager:

    def __init__(self):

        self.clients = {}

    ##########################################################

    def register(

            self,

            client_id,

            public_key

    ):

        self.clients[client_id] = public_key

    ##########################################################

    def authenticate(

            self,

            client_id,

            message,

            signature

    ):

        verifier = SignatureVerifier(

            self.clients[client_id]

        )

        return verifier.verify(

            message,

            signature

        )


##############################################################
# Example
##############################################################

if __name__ == "__main__":

    manager = KeyManager()

    private_key = manager.get_private_key()

    public_key = manager.get_public_key()

    message = "Federated Model Update"

    signer = DigitalSignature(

        private_key

    )

    signature = signer.sign(

        message

    )

    verifier = SignatureVerifier(

        public_key

    )

    print()

    print("Signature Valid :")

    print(

        verifier.verify(

            message,

            signature

        )

    )
    print("Create Clients -> Server -> Start Training")


"""
==============================================================
SPFI-HC

Part 7A

Hybrid Classification and Adaptive Fusion (HCAF)

Base Classifiers

==============================================================
"""

import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


##############################################################
# XGBoost Classifier
##############################################################

class XGBoostModel:

    def __init__(self):

        self.model = XGBClassifier(

            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42

        )

    def fit(self, X, y):

        self.model.fit(X, y)

    def predict(self, X):

        return self.model.predict(X)

    def predict_proba(self, X):

        return self.model.predict_proba(X)


##############################################################
# LightGBM Classifier
##############################################################

class LightGBMModel:

    def __init__(self):

        self.model = LGBMClassifier(

            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=8,
            random_state=42

        )

    def fit(self, X, y):

        self.model.fit(X, y)

    def predict(self, X):

        return self.model.predict(X)

    def predict_proba(self, X):

        return self.model.predict_proba(X)


##############################################################
# CatBoost Classifier
##############################################################

class CatBoostModel:

    def __init__(self):

        self.model = CatBoostClassifier(

            iterations=300,
            learning_rate=0.05,
            depth=8,
            loss_function="MultiClass",
            verbose=False,
            random_seed=42

        )

    def fit(self, X, y):

        self.model.fit(X, y)

    def predict(self, X):

        return self.model.predict(X).flatten()

    def predict_proba(self, X):

        return self.model.predict_proba(X)


##############################################################
# Hybrid Classifier
##############################################################

class HybridClassifier:

    def __init__(self):

        self.xgb = XGBoostModel()

        self.lgbm = LightGBMModel()

        self.cat = CatBoostModel()

    ##########################################################

    def fit(

            self,

            X,

            y

    ):

        print("Training XGBoost...")
        self.xgb.fit(X, y)

        print("Training LightGBM...")
        self.lgbm.fit(X, y)

        print("Training CatBoost...")
        self.cat.fit(X, y)

    ##########################################################

    def predict_probability(

            self,

            X

    ):

        p1 = self.xgb.predict_proba(X)

        p2 = self.lgbm.predict_proba(X)

        p3 = self.cat.predict_proba(X)

        return p1, p2, p3

    ##########################################################

    def predict(

            self,

            X

    ):

        p1, p2, p3 = self.predict_probability(X)

        probability = (

            p1 +

            p2 +

            p3

        ) / 3.0

        prediction = np.argmax(

            probability,

            axis=1

        )

        return prediction, probability


##############################################################
# Example
##############################################################

if __name__ == "__main__":

    from sklearn.datasets import make_classification

    X, y = make_classification(

        n_samples=1000,

        n_features=256,

        n_classes=5,

        n_informative=20,

        random_state=42

    )

    model = HybridClassifier()

    model.fit(X, y)

    pred, prob = model.predict(X)

    print()

    print("Prediction Shape :", pred.shape)

    print("Probability Shape :", prob.shape)

"""
==============================================================
SPFI-HC

Part 7B

Hybrid Classification and Adaptive Fusion (HCAF)

Adaptive Ensemble Fusion

==============================================================
"""

import numpy as np


##############################################################
# Confidence Calculator
##############################################################

class ConfidenceCalculator:

    def __init__(self):
        pass

    def confidence(self, probability):

        """
        Maximum class probability
        """

        return np.max(

            probability,

            axis=1

        )


##############################################################
# Adaptive Weight Estimation
##############################################################

class AdaptiveWeightEstimator:

    def __init__(self):
        pass

    def estimate(

            self,

            p1,

            p2,

            p3

    ):

        c1 = np.mean(

            np.max(p1, axis=1)

        )

        c2 = np.mean(

            np.max(p2, axis=1)

        )

        c3 = np.mean(

            np.max(p3, axis=1)

        )

        total = c1 + c2 + c3 + 1e-12

        w1 = c1 / total
        w2 = c2 / total
        w3 = c3 / total

        return w1, w2, w3


##############################################################
# Adaptive Probability Fusion
##############################################################

class AdaptiveFusion:

    def __init__(self):

        self.weight = AdaptiveWeightEstimator()

    ##########################################################

    def fuse(

            self,

            p1,

            p2,

            p3

    ):

        w1, w2, w3 = self.weight.estimate(

            p1,

            p2,

            p3

        )

        fused = (

            w1 * p1 +

            w2 * p2 +

            w3 * p3

        )

        prediction = np.argmax(

            fused,

            axis=1

        )

        confidence = np.max(

            fused,

            axis=1

        )

        return {

            "prediction": prediction,

            "probability": fused,

            "confidence": confidence,

            "weights": {

                "XGBoost": w1,

                "LightGBM": w2,

                "CatBoost": w3

            }

        }


##############################################################
# Performance Summary
##############################################################

class FusionStatistics:

    def summarize(

            self,

            result

    ):

        confidence = result["confidence"]

        return {

            "Average Confidence":

                float(np.mean(confidence)),

            "Maximum Confidence":

                float(np.max(confidence)),

            "Minimum Confidence":

                float(np.min(confidence)),

            "Confidence Std":

                float(np.std(confidence))

        }


##############################################################
# Complete Adaptive Fusion
##############################################################

class HCAF:

    """
    Hybrid Classification and Adaptive Fusion
    """

    def __init__(self):

        self.fusion = AdaptiveFusion()

        self.statistics = FusionStatistics()

    ##########################################################

    def predict(

            self,

            p1,

            p2,

            p3

    ):

        result = self.fusion.fuse(

            p1,

            p2,

            p3

        )

        report = self.statistics.summarize(

            result

        )

        result["statistics"] = report

        return result


##############################################################
# Example
##############################################################

if __name__ == "__main__":

    np.random.seed(42)

    p1 = np.random.rand(100,5)
    p2 = np.random.rand(100,5)
    p3 = np.random.rand(100,5)

    p1 = p1 / p1.sum(axis=1, keepdims=True)
    p2 = p2 / p2.sum(axis=1, keepdims=True)
    p3 = p3 / p3.sum(axis=1, keepdims=True)

    model = HCAF()

    output = model.predict(

        p1,

        p2,

        p3

    )

    print()

    print("Adaptive Weights")

    print(output["weights"])

    print()

    print("Prediction Shape")

    print(output["prediction"].shape)

    print()

    print(output["statistics"])
"""
==============================================================
SPFI-HC

Part 7C

Model Evaluation

Accuracy
Precision
Recall
F1-score
ROC-AUC
Confusion Matrix
Classification Report

==============================================================
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


##############################################################
# Performance Metrics
##############################################################

class PerformanceMetrics:

    def evaluate(
            self,
            y_true,
            y_pred,
            probability=None
    ):

        result = {}

        result["Accuracy"] = accuracy_score(
            y_true,
            y_pred
        )

        result["Precision"] = precision_score(
            y_true,
            y_pred,
            average="weighted"
        )

        result["Recall"] = recall_score(
            y_true,
            y_pred,
            average="weighted"
        )

        result["F1 Score"] = f1_score(
            y_true,
            y_pred,
            average="weighted"
        )

        if probability is not None:

            try:

                result["ROC AUC"] = roc_auc_score(
                    y_true,
                    probability,
                    multi_class="ovr"
                )

            except Exception:

                result["ROC AUC"] = None

        return result


##############################################################
# Confusion Matrix
##############################################################

class ConfusionMatrixGenerator:

    def generate(
            self,
            y_true,
            y_pred
    ):

        return confusion_matrix(
            y_true,
            y_pred
        )


##############################################################
# Classification Report
##############################################################

class ReportGenerator:

    def generate(
            self,
            y_true,
            y_pred
    ):

        return classification_report(
            y_true,
            y_pred,
            digits=4
        )


##############################################################
# Prediction Export
##############################################################

class PredictionExporter:

    def save(
            self,
            y_true,
            y_pred,
            filename="prediction.csv"
    ):

        dataframe = pd.DataFrame({

            "GroundTruth": y_true,

            "Prediction": y_pred

        })

        dataframe.to_csv(

            filename,

            index=False

        )

        print()

        print("Prediction Saved :", filename)


##############################################################
# Model Saver
##############################################################

class ModelSaver:

    def save(
            self,
            model,
            filename="best_model.pkl"
    ):

        joblib.dump(

            model,

            filename

        )

        print()

        print("Model Saved :", filename)


##############################################################
# Complete Evaluation
##############################################################

class Evaluator:

    def __init__(self):

        self.metric = PerformanceMetrics()

        self.matrix = ConfusionMatrixGenerator()

        self.report = ReportGenerator()

        self.export = PredictionExporter()

        self.saver = ModelSaver()

    ##########################################################

    def evaluate(
            self,
            model,
            y_true,
            y_pred,
            probability=None
    ):

        metrics = self.metric.evaluate(

            y_true,

            y_pred,

            probability

        )

        cm = self.matrix.generate(

            y_true,

            y_pred

        )

        report = self.report.generate(

            y_true,

            y_pred

        )

        print()

        print("=" * 60)

        print("Performance")

        print("=" * 60)

        for k, v in metrics.items():

            print(f"{k:15s}: {v}")

        print()

        print("=" * 60)

        print("Confusion Matrix")

        print("=" * 60)

        print(cm)

        print()

        print("=" * 60)

        print("Classification Report")

        print("=" * 60)

        print(report)

        self.export.save(

            y_true,

            y_pred

        )

        self.saver.save(

            model

        )

        return metrics


##############################################################
# Example
##############################################################

if __name__ == "__main__":

    np.random.seed(42)

    y_true = np.random.randint(

        0,

        5,

        1000

    )

    y_pred = np.random.randint(

        0,

        5,

        1000

    )

    probability = np.random.rand(

        1000,

        5

    )

    probability = probability / probability.sum(

        axis=1,

        keepdims=True

    )

    evaluator = Evaluator()

    evaluator.evaluate(

        model=None,

        y_true=y_true,

        y_pred=y_pred,

        probability=probability

    )

"""
==============================================================

SPFI-HC

Main Training Pipeline

==============================================================
"""

import torch
import numpy as np

# ==========================================================
# Data
# ==========================================================

from dataset.dataset_loader import MedicalDataset
from dataset.dataloader import get_dataloader

# ==========================================================
# Preprocessing
# ==========================================================

from preprocessing.qsammn import QSAMMN

# ==========================================================
# Feature Extraction
# ==========================================================

from models.swin_v2 import SwinFeatureExtractor
from models.mamba_encoder import MambaEncoder
from models.biomedlm_encoder import ClinicalTextEncoder
from models.ft_transformer import EHREncoder
from models.multimodal_fusion import MultimodalFusion

# ==========================================================
# Privacy
# ==========================================================

from privacy.msi_sva import MSISVA
from privacy.sensitivity_representation import SensitivityAwareFeature

from privacy.msi_fsa import MSIFSA
from privacy.binary_feature_selection import BinaryFeatureSelection
from privacy.feature_refinement import FeatureRefinement

# ==========================================================
# Blockchain
# ==========================================================

from blockchain.block import Blockchain

# ==========================================================
# Hybrid Classifier
# ==========================================================

from classification.hybrid_classifier import HybridClassifier
from classification.adaptive_fusion import HCAF
from classification.evaluation import Evaluator


##############################################################
# Pipeline
##############################################################

class SPFIHC:

    def __init__(self):

        ######################################################

        self.preprocessor = QSAMMN()

        ######################################################

        self.image_encoder = SwinFeatureExtractor()

        self.signal_encoder = MambaEncoder()

        self.text_encoder = ClinicalTextEncoder()

        self.ehr_encoder = EHREncoder()

        self.fusion = MultimodalFusion()

        ######################################################

        self.sva = MSISVA()

        self.sensitive = SensitivityAwareFeature()

        ######################################################

        self.fsa = MSIFSA()

        self.binary = BinaryFeatureSelection()

        self.refine = FeatureRefinement()

        ######################################################

        self.blockchain = Blockchain()

        ######################################################

        self.classifier = HybridClassifier()

        self.fusion_classifier = HCAF()

        ######################################################

        self.evaluator = Evaluator()

    ##########################################################

    def extract_feature(

            self,

            image,

            signal,

            text,

            ehr

    ):

        image = self.image_encoder(

            image

        )

        signal = self.signal_encoder(

            signal

        )

        text = self.text_encoder(

            text

        )

        ehr = self.ehr_encoder(

            ehr

        )

        feature = self.fusion(

            image,

            signal,

            text,

            ehr

        )

        return feature

    ##########################################################

    def privacy_feature(

            self,

            feature

    ):

        score = self.sva(

            feature

        )

        sensitive = self.sensitive(

            feature,

            score["privacy_score"]

        )

        swarm = self.fsa(

            sensitive["feature"],

            score["privacy_score"]

        )

        binary = self.binary(

            sensitive["feature"],

            swarm["best_position"]

        )

        refined = self.refine(

            binary["selected_feature"]

        )

        return refined["feature"]

    ##########################################################

    def train_classifier(

            self,

            X,

            y

    ):

        X = X.cpu().numpy()

        y = y.cpu().numpy()

        self.classifier.fit(

            X,

            y

        )

        p1, p2, p3 = self.classifier.predict_probability(

            X

        )

        result = self.fusion_classifier.predict(

            p1,

            p2,

            p3

        )

        metrics = self.evaluator.evaluate(

            self.classifier,

            y,

            result["prediction"],

            result["probability"]

        )

        return metrics


##############################################################
# Example
##############################################################

if __name__ == "__main__":

    print("="*60)

    print("SPFI-HC")

    print("Complete Framework")

    print("="*60)

    print()

    print("Modules Loaded Successfully")

    print()

    print("1. QSAMMN")

    print("2. Multimodal Learning")

    print("3. MSI-SVA")

    print("4. MSI-FSA")

    print("5. Blockchain")

    print("6. Federated Learning")

    print("7. Hybrid Classifier")

    print()

    print("Ready for Training...")
"""
==============================================================
SPFI-HC

Part 9A

Configuration File

==============================================================
"""

from dataclasses import dataclass


@dataclass
class DatasetConfig:

    image_size = 224

    batch_size = 16

    num_workers = 4

    train_ratio = 0.70

    validation_ratio = 0.15

    test_ratio = 0.15


############################################################


@dataclass
class ModelConfig:

    feature_dimension = 512

    num_classes = 5

    dropout = 0.30

    attention_heads = 8

    transformer_layers = 4


############################################################


@dataclass
class TrainingConfig:

    epochs = 100

    learning_rate = 1e-4

    weight_decay = 1e-5

    optimizer = "AdamW"

    scheduler = "CosineAnnealingLR"

    early_stopping = 15

    mixed_precision = True

    gradient_clip = 1.0


############################################################


@dataclass
class FederatedConfig:

    clients = 5

    communication_rounds = 20

    local_epochs = 2

    aggregation = "FedAvg"


############################################################


@dataclass
class BlockchainConfig:

    difficulty = 4

    hash_algorithm = "SHA256"

    signature = "ECDSA"


############################################################


@dataclass
class ExperimentConfig:

    random_seed = 42

    device = "cuda"

    save_directory = "./checkpoints"

    log_directory = "./logs"

    output_directory = "./results"


############################################################


class Config:

    dataset = DatasetConfig()

    model = ModelConfig()

    training = TrainingConfig()

    federated = FederatedConfig()

    blockchain = BlockchainConfig()

    experiment = ExperimentConfig()