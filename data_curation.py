import pandas as pd

DATASETS = ["PI.csv", "INI.csv", "NRTI.csv", "NNRTI.csv"]

def sequence_identity(seq1, seq2):
    length = min(len(seq1), len(seq2))
    matches = sum(seq1[i] == seq2[i] for i in range(length))
    return matches / length


for file_name in DATASETS:

    print("\nProcessing:", file_name)

    df = pd.read_csv(file_name, low_memory=False)
    print("Initial size:", len(df))

    # -----------------------------
    # Build sequence
    # -----------------------------
    seq_cols = [c for c in df.columns if c.startswith("P") and c[1:].isdigit()]
    seq_cols = sorted(seq_cols, key=lambda x: int(x[1:]))

    df[seq_cols] = df[seq_cols].astype(str)
    df["FullSeq"] = df[seq_cols].agg("".join, axis=1)

    # -----------------------------
    # Remove exact duplicates
    # -----------------------------
    df = df.drop_duplicates(subset=seq_cols).reset_index(drop=True)
    print("After exact dedup:", len(df))

    # -----------------------------
    # Remove rows with no resistance values
    # -----------------------------
    numeric_cols = df.select_dtypes(include="number").columns
    df = df.dropna(subset=numeric_cols, how="all")
    print("After phenotype filtering:", len(df))

    # -----------------------------
    # Identity filtering
    # -----------------------------
    if file_name != "INI.csv":   # Apply only to PI, NRTI, NNRTI

        threshold = 0.95
        sequences = df["FullSeq"].tolist()
        kept = []
        reps = []

        for i, seq in enumerate(sequences):
            redundant = False
            for r in reps:
                if sequence_identity(seq, r) >= threshold:
                    redundant = True
                    break
            if not redundant:
                reps.append(seq)
                kept.append(i)

        df_final = df.iloc[kept].reset_index(drop=True)

        # If <500 → try 90%
        if len(df_final) < 500:
            print("Re-running at 90% identity")
            threshold = 0.90
            kept = []
            reps = []

            for i, seq in enumerate(sequences):
                redundant = False
                for r in reps:
                    if sequence_identity(seq, r) >= threshold:
                        redundant = True
                        break
                if not redundant:
                    reps.append(seq)
                    kept.append(i)

            df_final = df.iloc[kept].reset_index(drop=True)

    else:
        print("Skipping identity filtering for INI (retain ≥500)")
        df_final = df.copy()

    print("Final size:", len(df_final))

    output = file_name.replace(".csv", "_FINAL.csv")
    df_final.to_csv(output, index=False)
    print("Saved:", output)

print("\nAll datasets processed.")
