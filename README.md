# 🔐 QKD Information Reconciliation using LDPC Codes  
This project demonstrates **LDPC-based Information Reconciliation** used in **Quantum Key Distribution (BB84)**.  
It simulates how Alice and Bob correct bit errors using **syndrome decoding + bit-flipping LDPC**.

---

## 🚀 Overview  
This code performs the full workflow of QKD error correction:

1. **Generate Alice's random key**
2. **Simulate noisy channel** → Bob receives corrupted key  
3. **Build an LDPC parity-check matrix (H)**
4. **Compute syndrome**: s = H × key (mod 2)
5. **Run bit-flipping decoder** to estimate error vector
6. **Bob corrects his key** to match Alice’s
7. **Prints statistics**:
   - Errors before correction  
   - Errors after correction  
   - Success / Failure  
   - Iterations used  
   - Syndrome match

This is a complete demonstration of **information reconciliation** in QKD.

---

## 📂 Repository Structure
```
qkd-ldpc-reconciliation/
│
├── src/
│   └── ldpc_reconciliation.py      # Main code
│
├── outputs/
│   ├── output_1.png                # Screenshot of program output
│   ├── output_2.png
│   └── ...
│
├── slides/
│   └── QKD_LDPC_Presentation.pdf   # PPT/PDF used for the project
│
└── README.md
```

---

## 🛠 Requirements
Install dependencies:

```
pip install numpy
```

That's all — the code uses only standard Python + NumPy.

---

## ▶ How to Run the Program
1. Open terminal  
2. Navigate to the project folder:

```
cd src
```

3. Run:

```
python ldpc_reconciliation.py
```

You will see:

- LDPC matrix construction  
- QBER estimate  
- Errors before correction  
- Errors after correction  
- Decoder iteration count  
- SUCCESS message if keys match  
- Syndrome match = True  

---

## 📸 Output Screenshots  
Screenshots of the terminal output are inside the **outputs/** folder.  
These serve as proof that the error-correction works correctly.

---

## 📊 Presentation Slides  
The presentation (PPT/PDF) used for this project is available in:

```
slides/QKD_LDPC_Presentation.pdf
```

---

## ✨ Author  
Mohammad Hasan & Team
(Team QKD Project Submission)

---

## 📘 Summary  
This repository provides a complete working demonstration of how **LDPC error-correction** is applied to **Quantum Key Distribution**.  
It shows the end-to-end flow:
**generate key → add noise → compute syndrome → decode → fix errors → match keys**.

This mirrors real BB84 post-processing steps.
