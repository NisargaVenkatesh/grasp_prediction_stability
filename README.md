# Grasp Stability Prediction (Vision-Based Tactile Sensing)

This repository contains a **Jupyter notebook** that trains and evaluates a model to **predict grasp stability** (stable vs. slipping/unstable) from **vision-based tactile sensing** data. Each grasp trial provides **two synchronized tactile RGB image streams** (left and right finger) over time, plus **binary stability labels per frame**. The core idea is: if the tactile images show deformation/pressure patterns that typically precede slip, a model can learn to flag “unstable” frames early.

Notebook:
- `Grasp_Stability_Prediction_Nisarga_Madhu_Venkatesh.ipynb`

---

## What the notebook does (end-to-end)

The notebook implements a full pipeline:

**1) Trial-aware data split (train/val/test).**  
Trials are split (not individual frames), so frames from the same grasp don’t leak into multiple splits. This matters because frames within one grasp are highly correlated—like consecutive frames in a video. If you split by frame, the model can “memorize” a trial’s appearance and your test score becomes misleading.

**2) Frame-level dataset creation.**  
From each trial, frames are sampled (optionally with a stride) and paired with per-frame labels. The notebook reports **frame-level metrics** as the primary results (accuracy/F1/AUC), because the number of trials is relatively small and trial-level metrics can have high variance.

**3) Vision backbone + classification head.**  
A pretrained image backbone (ConvNeXt) is used to extract features from tactile images, and then a small head predicts stability.

**4) Evaluation + visualizations.**  
The notebook saves:
- ROC curve
- confusion matrix
- a timeline plot for a sample trial showing predicted probabilities across frames

---

## Data format expected

The notebook expects a dataset directory structured as a set of trials, with each trial containing multiple HDF5 files. By default it looks here:

- `~/Downloads/grasp`

Per trial ID `{tid}`, expected files:

- `{tid}_tactileColorL.h5`  (left finger tactile RGB frames)
- `{tid}_tactileColorR.h5`  (right finger tactile RGB frames)
- `{tid}_label.h5`          (binary label per frame)
- `{tid}_gripForce.h5`      (optional, scalar per frame)
- `{tid}_normalForce.h5`    (optional, scalar per frame)

> Tip: If your dataset is stored elsewhere, change `DATA_ROOT` in the notebook to your path.

---

## Model approach (high level)

The notebook uses a **pretrained ConvNeXt** image model as a feature extractor. Think of ConvNeXt as a very strong “image pattern detector”: it turns each tactile RGB frame into a compact numeric representation (an embedding). The model then combines left/right embeddings (and optionally force features) and predicts a probability of stability for each frame.

Key toggles you can change inside the notebook:
- `BACKBONE = "facebook/convnext-tiny-224"`
- `USE_FORCES = False` (set `True` to include grip/normal force features)
- `FREEZE_BACKBONE = True` (freeze backbone for faster training and less overfitting)
- `FRAME_STRIDE = 5` (use every 5th frame; set `1` for all frames)

---

## Installation

The first cell installs dependencies:

```bash
pip install torch torchvision transformers h5py numpy scikit-learn tqdm pillow

```bash
Notes:

PyTorch is the deep learning framework used to train the model.

Transformers (Hugging Face) is used here to load the ConvNeXt backbone and image processor.

h5py reads .h5 tactile/label/force files.

How to run

Clone the repository:

git clone https://github.com/NisargaVenkatesh/grasp_prediction_stability.git
cd grasp_prediction_stability

Open the notebook:

jupyter notebook

Then run Grasp_Stability_Prediction_Nisarga_Madhu_Venkatesh.ipynb.

Ensure your data path is correct:

Update DATA_ROOT if needed (Data is large and restricted to the Chair of TU Dresden therefore I have not included it in the repo)

Run all cells to train and evaluate.

Outputs are saved (by default) to:

~/Downloads/runs_grasp/

Why frame-level metrics?

In tactile grasp stability data, you often have many frames per trial but far fewer trials. Trial-level evaluation can swing a lot if you only have ~10 test trials. Frame-level evaluation answers a different (but very practical) question:
“At any moment during the grasp, can we correctly detect stability/slip from tactile feedback?”
That’s especially relevant if you want to trigger corrective actions in real time (e.g., increase grip force, adjust pose).

License / usage

This repo is intended for academic/coursework and experimentation. If you reuse parts of the notebook, please attribute appropriately.

Contact

Author: Nisarga Madhu Venkatesh
