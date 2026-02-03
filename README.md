# 📱 Edge AI Sentinel: Real-Time Intrusion Detection on Android

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/Platform-Android%20%7C%20Termux-green)
![Library](https://img.shields.io/badge/Library-River%20%7C%20PyTorch-orange)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-red)

**Can we defend IoT devices using only their own limited CPU?**
This project benchmarks "Green AI" algorithms (Online Statistical Learning) against traditional Deep Learning for cybersecurity, running entirely on a standard smartphone (ARM64 architecture) via Termux.

---

## 🚀 The Experiment
Most "Edge AI" research runs simulations on powerful laptops. **This project runs on the actual edge.**

* **Device:** Samsung Galaxy [Your Model]
* **Environment:** Ubuntu 25.10 (via Proot/Termux)
* **Dataset:** NSL-KDD (Network Intrusion Traffic)
* **Constraint:** No GPU, Limited RAM (<512MB allowed)

### 📸 Proof of Concept (Live Execution)
*[Insert your "Hacker Shot" screenshot here - showing the terminal text scrolling]*
> *Figure 1: The system processing network packets in real-time on Android.*

---

## 📊 Results: Deep Learning vs. Hoeffding Trees

We compared three models processing a stream of network traffic. The goal: High accuracy with minimal battery/RAM usage.

| Model | Accuracy | Inference Speed | RAM Usage | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Hoeffding Adaptive Tree (Ours)** | **96.8%** | **~0.2 ms** | **~210 MB** | 🏆 **Best for Edge** |
| Adaptive Random Forest | 96.9% | ~0.5 ms | >300 MB | Accurate but Heavy |
| Online MLP (Deep Learning) | 94.5% | >1.5 ms | ~300 MB | Slow & Complex |

### 📈 Performance Graphs
*[Insert 'compare_accuracy.png' here]*
> *The Hoeffding Tree (Green) converges as fast as the heavy Random Forest.*

*[Insert 'compare_latency.png' here]*
> *Deep Learning (Red) is 7x slower per packet than our Statistical model.*

---

## 🛠️ Installation & Usage

Want to try this on your own phone? You only need [Termux](https://termux.dev).

**1. Setup Environment**
```bash
pkg install proot-distro python git
proot-distro install ubuntu
proot-distro login ubuntu
