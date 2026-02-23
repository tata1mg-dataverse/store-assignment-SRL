# Store Assignment for Order Fulfilment in an E-Commerce Supply Chain A Deep Reinforcement Learning Based Approach

> A learning-based system for real-time store assignment in an e-pharmacy, combining **imitation learning** and **reinforcement learning** to optimise delivery time and cost under regulatory constraints — achieving a **5% cost reduction at scale**.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Delivery Classification](#-delivery-classification)
- [Training](#-training)

---

## 🔍 Overview

In an e-pharmacy supply chain, each customer order must be assigned to a fulfilling store in real time. Poor assignment decisions lead to delayed deliveries and inflated logistics costs. This project frames store assignment as a sequential decision-making problem and solves it using a two-stage learning pipeline:

1. **Supervised / Imitation Learning** — a policy network is pre-trained on historical expert assignments to bootstrap performance.
2. **Reinforcement Learning** — the pre-trained policy is fine-tuned using RL to optimise long-term objectives (delivery speed + cost), subject to regulatory constraints.

The system operates at scale across thousands of daily orders, balancing fast and slow delivery SLAs across geographies.

---

## 📦 Dataset

The dataset contains order-level delivery information used for the Store Allocation project. Following provides definition of few fundamental keywords used in the dataset.

### 1. Orders

Core order metadata.

| Column | Description |
|---|---|
| `GROUP_ID` | Parent order ID. Multiple individual orders may be grouped under a single `GROUP_ID`. |
| `ORDER_ID` | Unique identifier for each individual order. |
| `CENTRAL_PINCODE` | Pincode of the customer's delivery location. Used to determine eligible fulfillment stores. |

### 2. Order SKUs

Line-item details for products within each order.

| Column | Description |
|---|---|
| `SKU_ID` | Unique identifier for the ordered product. |
| `UNITS_SOLD` | Number of units of the product ordered. |

### 3. ETA / TAT Quantiles

Delivery time estimates and actuals, used as targets and features during training.

| Column | Description |
|---|---|
| `DELIVERY_TYPE` | Type of delivery — `fast` (minutes to hours) or `slow` (1+ days). |
| `TAT` | **Turn Around Time** — actual time taken to deliver the product to the customer. Used as the ground truth label. |
| `ETA` | **Estimated Time of Arrival** — delivery time estimate shown to the customer at order placement. |

---

## 🚚 Delivery Classification

Delivery speed is classified based on **TAT**:

| Type | Condition | Description |
|---|---|---|
| ⚡ **Fast Delivery** | TAT < 1 day | Delivered within minutes to hours |
| 🕐 **Slow Delivery** | TAT ≥ 1 day | Standard next-day or longer delivery |

The `fast_delivery_alpha` training parameter controls how much weight the model places on optimising fast delivery assignments relative to slow ones.

---

## 🏋️ Training

Training follows a two-stage process. Always train the supervised model first — its output is used to initialise the RL policy.

### Stage 1 — Supervised (Imitation Learning)

Pre-trains the policy network on historical store assignment decisions.

```bash
python run_supervised.py \
  --epoch_size 200 \
  --learning_rate 0.004 \
  --alpha 0.25 \
  --fast_delivery_alpha 0.75
```

### Stage 2 — Reinforcement Learning

Fine-tunes the pre-trained policy using RL. Requires the saved model checkpoints from Stage 1.

```bash
python run_rl.py \
  --epoch_size 200 \
  --learning_rate 0.004 \
  --alpha 0.25 \
  --fast_delivery_alpha 0.75 \
  --policy_model_location 'policy_model.pt' \
  --value_model_location 'value_model.pt'
```

---