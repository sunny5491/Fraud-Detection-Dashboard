"""
Synthetic e-commerce returns dataset generator.

Produces data/processed/returns_fraud_dataset.csv (same schema the pipeline consumes)
and data/processed/fraud_labels.csv (ground-truth persona labels, used ONLY for
offline evaluation — the unsupervised model never sees them).

Design goals:
- Users have realistic multi-order histories (the behavioral features need history
  to be meaningful: return frequency, reason diversity, days active, etc.).
- Return dates are always on or after the purchase date.
- ~3% of users follow one of three fraud personas with distinct behavioral
  signatures, giving the evaluation notebook ground truth to score against.

Run from the repo root:  python3 src/generate_dataset.py
"""
import os
import numpy as np
import pandas as pd

SEED = 42
N_USERS = 1500
FRAUD_RATE = 0.03
START = pd.Timestamp("2023-01-01")
END = pd.Timestamp("2024-12-31")

RETURN_REASONS = ["Defective", "Wrong item", "Changed mind", "Not as described"]
PAYMENT_METHODS = ["Credit Card", "Debit Card", "PayPal", "Gift Card"]

rng = np.random.default_rng(SEED)


def sample_price(high_value_bias: float = 0.0) -> float:
    """Lognormal item price; high_value_bias in [0,1] shifts mass above $500."""
    if rng.random() < high_value_bias:
        return float(np.round(rng.uniform(500, 2000), 2))
    return float(np.round(min(np.exp(rng.normal(4.6, 0.8)), 1999.0), 2))


def make_user(user_idx: int, persona: str) -> list[dict]:
    """Generate all transactions for one user according to their persona."""
    user_id = f"USER{user_idx:08d}"
    signup = START + pd.Timedelta(days=int(rng.integers(0, 550)))
    horizon_days = max((END - signup).days, 30)

    if persona == "normal":
        n_orders = int(np.clip(rng.negative_binomial(2, 0.25) + 1, 1, 40))
        p_return = float(np.clip(rng.beta(2, 8), 0.02, 0.6))
        high_value_bias = 0.05
        return_delay = lambda: int(rng.integers(2, 31))
        reason_weights = np.array([0.3, 0.25, 0.25, 0.2])
    elif persona == "serial_returner":
        # Returns most of what they buy, across many orders.
        n_orders = int(rng.integers(15, 41))
        p_return = float(rng.uniform(0.70, 0.95))
        high_value_bias = 0.10
        return_delay = lambda: int(rng.integers(2, 25))
        reason_weights = np.array([0.25, 0.25, 0.3, 0.2])
    elif persona == "high_value_abuser":
        # Targets expensive items and refunds them, often claiming defects.
        n_orders = int(rng.integers(6, 20))
        p_return = float(rng.uniform(0.55, 0.85))
        high_value_bias = 0.75
        return_delay = lambda: int(rng.integers(3, 28))
        reason_weights = np.array([0.55, 0.15, 0.1, 0.2])
    elif persona == "wardrober":
        # "Free renting": rapid-fire purchases returned quickly with the same excuse.
        n_orders = int(rng.integers(8, 25))
        p_return = float(rng.uniform(0.60, 0.90))
        high_value_bias = 0.25
        return_delay = lambda: int(rng.integers(1, 6))
        reason_weights = np.array([0.05, 0.05, 0.75, 0.15])
    else:
        raise ValueError(persona)

    rows = []
    order_days = np.sort(rng.integers(0, horizon_days, size=n_orders))
    for j, day in enumerate(order_days):
        purchase_date = signup + pd.Timedelta(days=int(day))
        price = sample_price(high_value_bias)
        returned = rng.random() < p_return
        if returned:
            return_date = purchase_date + pd.Timedelta(days=return_delay())
            reason = str(rng.choice(RETURN_REASONS, p=reason_weights / reason_weights.sum()))
        else:
            return_date, reason = pd.NaT, "Not Returned"
        order_id = f"ORD{user_idx:05d}{j:03d}"
        item_id = f"PROD{int(rng.integers(0, 5000)):08d}"
        rows.append({
            "user_id": user_id,
            "order_id": order_id,
            "item_id": item_id,
            "purchase_date": purchase_date.date(),
            "return_date": return_date.date() if returned else "",
            "return_reason": reason,
            "refund_amount": price,
            "payment_method": str(rng.choice(PAYMENT_METHODS, p=[0.45, 0.3, 0.2, 0.05])),
            "receipt_id": f"{order_id}-{item_id}",
        })
    return rows


def main() -> None:
    n_fraud = int(N_USERS * FRAUD_RATE)
    personas = (["normal"] * (N_USERS - n_fraud)
                + list(rng.choice(["serial_returner", "high_value_abuser", "wardrober"],
                                  size=n_fraud)))
    rng.shuffle(personas)

    all_rows, labels = [], []
    for i, persona in enumerate(personas):
        all_rows.extend(make_user(i, persona))
        labels.append({"user_id": f"USER{i:08d}",
                       "is_fraud": int(persona != "normal"),
                       "persona": persona})

    df = pd.DataFrame(all_rows)
    labels_df = pd.DataFrame(labels)

    out_dir = os.path.join("data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "returns_fraud_dataset.csv"), index=False)
    labels_df.to_csv(os.path.join(out_dir, "fraud_labels.csv"), index=False)

    n_returns = (df["return_reason"] != "Not Returned").sum()
    print(f"users: {N_USERS} ({n_fraud} fraud) | transactions: {len(df)} | returns: {n_returns}")
    print(labels_df[labels_df.is_fraud == 1].persona.value_counts().to_string())


if __name__ == "__main__":
    main()
