import pandas as pd
import numpy as np
import random
from itertools import combinations

# --------------------------------------------
# CONFIGURATION
# --------------------------------------------
DATASETS = ["PI.csv", "INI.csv", "NRTI.csv", "NNRTI.csv"]
IDENTITY_THRESHOLD = 0.95
FALLBACK_THRESHOLD = 0.90
IDENTITY_SAMPLE_SIZE = 1000


# --------------------------------------------
# SEQUENCE IDENTITY (STRICT ALIGNMENT)
# --------------------------------------------
def sequence_identity(seq1, seq2):
    if len(seq1) != len(seq2):
        raise ValueError("Aligned sequences must have equal length.")
    matches = sum(a == b for a, b in zip(seq1, seq2))
    return matches / len(seq1)


# --------------------------------------------
# SAMPLE-BASED IDENTITY STATISTICS
# --------------------------------------------
def estimate_identity_stats(sequences, threshold, sample_size):

    if len(sequences) < 2:
        return 0, 0

    pairs = list(combinations(range(len(sequences)), 2))

    if len(pairs) > sample_size:
        pairs = random.sample(pairs, sample_size)

    identities = []
    high_identity_count = 0

    for i, j in pairs:
        id_val = sequence_identity(sequences[i], sequences[j])
        identities.append(id_val)
        if id_val >= threshold:
            high_identity_count += 1

    mean_identity = np.mean(identities)
    high_identity_percent = 100 * high_identity_count / len(pairs)

    return mean_identity, high_identity_percent


# --------------------------------------------
# GREEDY IDENTITY CLUSTERING
# --------------------------------------------
def greedy_identity_clustering(df, threshold):
    sequences = df["FullSeq"].tolist()
    representatives = []
    keep_indices = []

    for i, seq in enumerate(sequences):
        redundant = False
        for rep in representatives:
            if sequence_identity(seq, rep) >= threshold:
                redundant = True
                break
        if not redundant:
            representatives.append(seq)
            keep_indices.append(i)

    return df.iloc[keep_indices].reset_index(drop=True)


# --------------------------------------------
# BUILD CONSENSUS
# --------------------------------------------
def build_consensus(df, seq_cols):
    consensus = ""
    for col in seq_cols:
        consensus += df[col].mode()[0]
    return consensus


# --------------------------------------------
# EXTRACT TRUE MUTATIONS (REFERENCE-BASED)
# --------------------------------------------
def extract_mutations(seq, consensus):
    mutations = []
    for i, (aa, ref) in enumerate(zip(seq, consensus)):
        if aa != ref:
            mutations.append(f"{ref}{i+1}{aa}")
    return mutations


# --------------------------------------------
# MAIN PROCESSING
# --------------------------------------------
for file_name in DATASETS:

    print("\n========================================")
    print("Processing:", file_name)

    df = pd.read_csv(file_name, low_memory=False)
    initial_size = len(df)
    print("Initial dataset size:", initial_size)

    # --------------------------------------------
    # Identify sequence columns
    # --------------------------------------------
    seq_cols = [c for c in df.columns if c.startswith("P") and c[1:].isdigit()]
    seq_cols = sorted(seq_cols, key=lambda x: int(x[1:]))

    print("Protein alignment length:", len(seq_cols))

    # --------------------------------------------
    # CLEAN ALIGNMENT COLUMNS SAFELY
    # --------------------------------------------
    df[seq_cols] = df[seq_cols].astype(str)

    # Strip whitespace
    df[seq_cols] = df[seq_cols].apply(lambda col: col.str.strip())

    # If multi-letter (e.g., LM, GE), keep first letter only
    df[seq_cols] = df[seq_cols].apply(lambda col: col.str[0])

    # --------------------------------------------
    # BUILD FULL SEQUENCE
    # --------------------------------------------
    df["FullSeq"] = df[seq_cols].agg("".join, axis=1)

    # Validate uniform length
    lengths = df["FullSeq"].apply(len)
    print("Min length:", lengths.min())
    print("Max length:", lengths.max())

    if lengths.nunique() != 1:
        raise ValueError("Sequences are not uniformly aligned.")

    print("Confirmed uniform sequence length:", lengths.iloc[0])

    # --------------------------------------------
    # REMOVE EXACT DUPLICATES
    # --------------------------------------------
    df = df.drop_duplicates(subset=seq_cols).reset_index(drop=True)
    print("After exact duplicate removal:", len(df))

    # --------------------------------------------
    # REMOVE ROWS WITHOUT PHENOTYPE
    # --------------------------------------------
    numeric_cols = df.select_dtypes(include="number").columns
    df = df.dropna(subset=numeric_cols, how="all")
    print("After phenotype filtering:", len(df))
    print("Drug resistance columns:", list(numeric_cols))

    # --------------------------------------------
    # IDENTITY STATS BEFORE FILTERING
    # --------------------------------------------
    sequences_before = df["FullSeq"].tolist()
    mean_before, high_before = estimate_identity_stats(
        sequences_before,
        IDENTITY_THRESHOLD,
        IDENTITY_SAMPLE_SIZE
    )

    print(f"Mean identity BEFORE filtering: {mean_before:.4f}")
    print(f"% pairs ≥ {IDENTITY_THRESHOLD*100:.0f}% BEFORE: {high_before:.2f}%")

    # --------------------------------------------
    # GREEDY IDENTITY FILTERING
    # --------------------------------------------
    if file_name != "INI.csv":

        print(f"Applying greedy clustering at {IDENTITY_THRESHOLD*100}%")

        df_final = greedy_identity_clustering(df, IDENTITY_THRESHOLD)

        if len(df_final) < 500:
            print("Dataset <500. Re-running at 90%.")
            df_final = greedy_identity_clustering(df, FALLBACK_THRESHOLD)

    else:
        print("Skipping identity filtering for INI to preserve ≥500 sequences.")
        df_final = df.copy()

    final_size = len(df_final)
    reduction_percent = 100 * (1 - final_size / initial_size)

    print("Final dataset size:", final_size)
    print(f"Redundancy reduced by: {reduction_percent:.2f}%")

    # --------------------------------------------
    # IDENTITY STATS AFTER FILTERING
    # --------------------------------------------
    sequences_after = df_final["FullSeq"].tolist()
    mean_after, high_after = estimate_identity_stats(
        sequences_after,
        IDENTITY_THRESHOLD,
        IDENTITY_SAMPLE_SIZE
    )

    print(f"Mean identity AFTER filtering: {mean_after:.4f}")
    print(f"% pairs ≥ {IDENTITY_THRESHOLD*100:.0f}% AFTER: {high_after:.2f}%")

    # --------------------------------------------
    # CONSENSUS & MUTATION LIST
    # --------------------------------------------
    consensus = build_consensus(df_final, seq_cols)

    df_final["MutationList"] = df_final["FullSeq"].apply(
        lambda seq: extract_mutations(seq, consensus)
    )

    # --------------------------------------------
    # BIOLOGICAL TARGET VALIDATION
    # --------------------------------------------
    if "PI" in file_name:
        print("Target protein: Protease (~99 aa expected)")
    elif "NRTI" in file_name or "NNRTI" in file_name:
        print("Target protein: Reverse Transcriptase (~560 aa expected)")
    elif "INI" in file_name:
        print("Target protein: Integrase (~288 aa expected)")

    # --------------------------------------------
    # SAVE FINAL DATASET
    # --------------------------------------------
    output = file_name.replace(".csv", "_FINAL.csv")
    df_final.to_csv(output, index=False)
    print("Saved:", output)

print("\nAll datasets processed successfully.")