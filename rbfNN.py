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
        X_norm = np.sum(X**2, axis=1).reshape(-1, 1)
        C_norm = np.sum(self.centers**2, axis=1).reshape(1, -1)
        dists = X_norm + C_norm-2*np.dot(X, self.centers.T)
        return np.exp(-dists/(2*(self.sigma**2)))
    
    def fit(self, X, y):
        kmeans = KMeans(n_clusters=self.n_centers, random_state=42, n_init=10)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_

        G = self._calculate_phi(X)

        self.weights = np.linalg.pinv(G)@y
    
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

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def show_images(indices, title, n=5):
    plt.figure(figsize=(15, 4))
    n = min(n, len(indices))
    for i, idx in enumerate(indices[:n]):
        img = test_imgs[idx]
        label = y_test[idx]
        img = img*torch.tensor((0.2470, 0.2435, 0.2616)).view(3,1,1)
        img = img+torch.tensor((0.4914, 0.4822, 0.4465)).view(3,1,1)
        img = img.permute(1,2,0).numpy()

        plt.subplot(1, n, i+1)
        plt.imshow(img)
        plt.axis("off")

        true_label = class_names[label]
        pred_label_val = y_test_pred[idx]
        pred_label = "cat" if pred_label_val==0 else "dog"
        plt.title(f"True: {true_label}\nPred: {pred_label}")

    plt.suptitle(title)
    plt.show()
    

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


centers_list = [10, 50, 100, 200, 500]
sigma_list = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0]
results = []
for i in centers_list:
    for j in sigma_list:
        rbf = RBFNN(n_centers=i, sigma=j)
        start = time.perf_counter()
        rbf.fit(X_train_pca, y_train_binary)
        end = time.perf_counter()
        train_time = end-start

        y_train_pred = rbf.predict(X_train_pca)
        y_test_pred = rbf.predict(X_test_pca)
        train_accuracy = accuracy_score(y_train_binary, y_train_pred)
        test_accuracy = accuracy_score(t_test_binary, y_test_pred)

        results.append((i, j, train_accuracy, test_accuracy, train_time))
        print(f"Centers={i}, Sigma={j}")
        print(f"Train Accuracy: {train_accuracy*100:.2f}%")
        print(f"Test Accuracy: {test_accuracy*100:.2f}%")
        print(f"Train Time: {train_time:.2f} sec")

correct_idx = np.where(t_test_binary==y_test_pred)[0]
wrong_idx = np.where(t_test_binary!=y_test_pred)[0]
show_images(correct_idx, "Correctly Classified Images")
show_images(wrong_idx, "Misclassified Images")