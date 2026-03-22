# Proiect: Logistic Regression – CPU vs GPU vs CUDA

## 1. Cerințele Proiectului

### Obiectiv
Scopul proiectului este implementarea, profilarea și compararea a trei versiuni distincte ale algoritmului de Machine Learning **Logistic Regression** pentru o problemă de clasificare binară, utilizând un set de date de mari dimensiuni.

Obiectivul principal este analizarea:
- performanței (timpului de antrenare)
- scalabilității algoritmului

prin utilizarea paralelizării masive pe acceleratoare grafice (GPU).

---

## 2. Implementările realizate

### 🔹 Baseline (Secvențial / CPU)
- Implementare standard rulată pe procesor
- Folosește biblioteca **Scikit-Learn**
- Servește ca punct de referință pentru comparație

---

### 🔹 Versiunea GPU (Framework)
- Implementare accelerată hardware
- Utilizează ecosistemul **RAPIDS cuML**
- Beneficiază de optimizări GPU fără implementare manuală

---

### 🔹 Implementare CUDA (Custom)
- Implementare manuală a algoritmului
- Paralelizarea Gradient Descent folosind kernel-uri CUDA
- Scrisă în **CUDA C/C++**
- Control complet asupra execuției pe GPU

---

## 3. Tehnologii și Limbaje utilizate

### Limbaje de programare:
- Python
- C++
- CUDA C

### Framework-uri / Biblioteci:
- Scikit-Learn
- NumPy
- Pandas
- RAPIDS cuML
- PyCUDA

### Ecosistem Hardware:
- NVIDIA CUDA Toolkit

---

## 4. Informații despre mașina de rulare (Hardware & OS)

Testele au fost rulate pe următoarea configurație:

- **Sistem de operare:** Windows 11 + WSL2 (Ubuntu 24.04 LTS)
- **Procesor (CPU):** AMD Ryzen 7 5700
- **Memorie RAM:** 32 GB
- **Placă Video (GPU):** NVIDIA GeForce GTX 1660
- **Memorie Video (VRAM):** 6GB
- **Versiune Python:** 3.12 (mediu virtual izolat)

---

## 5. Rezulate experimentale

Testele de performanță se realizează pe dataset-uri de dimensiuni mari (ex. 500.000 instanțe, 30 caracteristici). Metrica principală de comparație este timpul de execuție necesar pentru antrenarea modelului, excluzând timpul de preprocesare a datelor.


| Versiune Algoritm          | Hardware Utilizat | Timp de antrenare (Secunde) | Acuratețe Model (%) |
| -------------------------- | ----------------- | --------------------------- | ------------------- |
| 1. Baseline (Scikit-Learn) | CPU               | 0.4482 s                    | 99.91%              |


## 5. Dataset

Acest proiect utilizează setul de date **Credit Card Fraud Detection** pentru a testa performanța algoritmilor. Deoarece fișierul depășește limita de dimensiune admisă de GitHub (150+ MB), acesta nu este inclus direct în repository.

**Instrucțiuni pentru a rula codul local:**
1. Descărcați dataset-ul oficial de pe Kaggle: [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Dezarhivați fișierul descărcat.
3. Plasați fișierul `creditcard.csv` exact în directorul rădăcină al acestui proiect (la același nivel cu fișierele `.py`).