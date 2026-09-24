# Binary Image Classification on CIFAR-10 using RBFNN & Baseline Classifiers

Αυτό το αποθετήριο περιέχει την υλοποίηση και πειραματική αξιολόγηση ενός **Radial Basis Function Neural Network (RBFNN)** από το μηδέν (from scratch), με σκοπό τη δυαδική ταξινόμηση εικόνων (Γάτες vs Σκύλοι) από το σύνολο δεδομένων **CIFAR-10**. 

Το μοντέλο συγκρίνεται επίσης με κλασικούς αλγορίθμους μηχανικής μάθησης (**k-NN** και **Nearest Centroid**).

---

## 🛠️ Pipeline & Μεθοδολογία

1. **Σύνολο Δεδομένων (Dataset):**
   * Χρήση του **CIFAR-10** μέσω `torchvision.datasets`.
   * Φιλτράρισμα για εξαγωγή των κλάσεων: `Cat (3)` και `Dog (5)` και μετατροπή των labels σε δυαδική μορφή (0 και 1).

2. **Προεπεξεργασία & Μείωση Διαστατικότητας:**
   * Κανονικοποίηση (Standardization) των δεδομένων.
   * Εφαρμογή **PCA (Principal Component Analysis)** για επιλογή του αριθμού συνιστωσών που διατηρεί τουλάχιστον το **90%** της συνολικής διασποράς (variance).
   * Κλιμάκωση χαρακτηριστικών μέσω `StandardScaler`.

3. **Υλοποίηση Custom RBF Neural Network:**
   * **Hidden Layer:** Εύρεση των κέντρων Gaussian RBF με χρήση του αλγορίθμου **K-Means**.
   * **Output Layer:** Υπολογισμός βαρών κλειστής μορφής μέσω ψευδοαντιστρόφου πίνακα (Moore-Penrose pseudo-inverse).
   * **Grid Search:** Δοκιμή διαφορετικών συνδυασμών αριθμού κέντρων (`centers`) και παραμέτρου διασποράς (`sigma`).

4. **Συγκριτική Αξιολόγηση (Benchmarking):**
   * Σύγκριση ακρίβειας (Accuracy) και χρόνου εκπαίδευσης/πρόβλεψης με:
     * **1-Nearest Neighbor (1-NN)**
     * **3-Nearest Neighbors (3-NN)**
     * **Nearest Centroid Classifier**

5. **Οπτικοποίηση:**
   * Ανάλυση σφαλμάτων και οπτικοποίηση σωστά και λανθασμένα ταξινομημένων εικόνων (True vs Predicted label).

---

## 💻 Τεχνολογίες & Βιβλιοθήκες

* **Python 3**
* **PyTorch & Torchvision** (Φόρτωση και επεξεργασία CIFAR-10)
* **NumPy** (Vectorized υπολογισμοί και linear algebra)
* **Scikit-learn** (PCA, StandardScaler, K-Means, k-NN, Nearest Centroid, Metrics)
* **Matplotlib** (Οπτικοποίηση εικόνων και αποτελεσμάτων)

---

## 🚀 Εκτέλεση

```bash
# Εγκατάσταση απαραίτητων βιβλιοθηκών
pip install torch torchvision numpy scikit-learn matplotlib

# Εκτέλεση του script
python main.py
