import torch
from sklearn.preprocessing import StandardScaler
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class RBFNN:
    def __init__(self, n_centers=100, sigma=1.0):
        self.n_centers = n_centers
        self.sigma = sigma
        self.centers = None
        self.weights = None
    
    def _gaussian_rbf(self, x, center):
        distance = np.linalg.norm(x-center)
        return np.exp(-(distance**2) / (2*(self.sigma**2)))

    def  _calculate_phi(self, X):
        G = np.zeros((X.shape[0], self.n_centers))
        for i,x in enumerate(X):
            for j,c in enumerate(self.centers):
                G[i,j] = self._gaussian_rbf(x, c)
        return G
    
    def fit(self, X, y):
        print(f"Εκπαίδευση K-Means για {self.n_centers} κέντρα")
        kmeans = KMeans(n_clusters=self.n_centers, random_state=42, n_init=10)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_

        G = self._calculate_phi(X)

        self.weights = np.linalg.pinv(G)@y
        print("Η εκπαίδευση ολοκληρώθηκε")
    
    def predict(self, X):
        G = self._calculate_phi(X)
        predictions = G @ self.weights
        return np.where(predictions>=0.5, 1, 0)



def extract_classes(dataset, classes):
    X = []
    y = []
    imgs = [] 
    for img, label in dataset:
        if label in classes:
            img_flat = img.view(-1).numpy()
            imgs.append(img)
            X.append(img_flat)
            y.append(label)
    X = np.array(X)
    y = np.array(y)
    return X, y, imgs

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

classes_to_keep = [3, 5]
X_train, y_train, _ = extract_classes(train_dataset, classes_to_keep)
X_test, y_test, test_imgs = extract_classes(test_dataset, classes_to_keep)

pca_temp = PCA()
pca_temp.fit(X_train)
explained_variance = np.cumsum(pca_temp.explained_variance_ratio_)
n_components = np.argmax(explained_variance >= 0.90) + 1
print("Αριθμός συνιστωσών για 90% πληροφορίας:", n_components)

pca = PCA(n_components=n_components)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("Train PCA shape:", X_train_pca.shape)
print("Test PCA shape:", X_test_pca.shape)

scaler = StandardScaler()
X_train_pca = scaler.fit_transform(X_train_pca)
X_test_pca = scaler.transform(X_test_pca)

y_train_binary = np.where(y_train == 3, 0, 1)
t_test_binary = np.where(y_test == 3, 0, 1)

rbf = RBFNN(n_centers=100, sigma=2.0)
rbf.fit(X_train_pca, y_train_binary)