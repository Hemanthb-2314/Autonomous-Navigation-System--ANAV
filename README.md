# 🛰️ ANAV Terrain Classifier – YOLOv8 for Mars Navigation

This repository contains the machine learning component of the **Autonomous Aerial Navigation System for Mars Exploration (ANAV)** – a mini project completed at Ramaiah Institute of Technology. This module focuses on training and deploying a **YOLOv8-based deep learning model** to classify Martian terrain into **safe**, **rocky**, and **transitional** zones to assist autonomous UAVs in decision-making during planetary exploration.

---

## 🚀 Project Context

Due to the absence of GPS on Mars and the harsh terrain, aerial robots need intelligent systems to autonomously identify safe landing zones and plan paths in real-time. This terrain classification module was designed to:
- Support **GNSS-denied navigation**
- Work with **visual-inertial odometry systems**
- Enable autonomous exploration, landing, and return

---

## 🧠 Model Overview

- **Model**: YOLOv8 (fine-tuned from COCO weights)
- **Classes**: `safe`, `rocky`, `transitional`
- **Input size**: 640×640
- **Training time**: 300 epochs
- **Optimization**: TensorRT, FP16, Pruning, Knowledge Distillation


