import pandas as pd
import numpy as np
import random
from itertools import combinations

# --------------------------------------------
# CONFIGURATION
# --------------------------------------------
DATASETS = ["PI.csv", "INI.csv", "NRTI.csv", "NNRTI.csv"]
INITIAL_THRESHOLD = 0.95
MAX_THRESHOLD = 0.99
IDENTITY_SAMPLE_SIZE = 1000


# --------------------------------------------
# SEQUENCE IDENTITY (STRICT)
# --------------------------------------------
def sequence_identity(seq1, seq2):
    if len(seq1) != len(seq2):
        raise ValueError("Aligned sequences must have equal length.")
    matches = sum(a == b for a, b in zip(seq1, seq2))
    return matches / len(seq1)


# --------------------------------------------
# SAMPLE IDENTITY STATISTICS
# --------------------------------------------
def estimate_identity_stats(sequences, threshold, sample_size):
    if len(sequences) < 2:
        return 0, 0

    pairs = list(combinations(range(len(sequences)), 2))
    if len(pairs) > sample_size:
        pairs = random.sample(pairs, sample_size)

    identities = []
    high_identity = 0

    for i, j in pairs:
        val = sequence_identity(sequences[i], sequences[j])
        identities.append(val)
        if val >= threshold:
            high_identity += 1

    return np.mean(identities), 100 * high_identity / len(pairs)


# --------------------------------------------
# GREEDY CLUSTERING
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
# BUILD CONSENSUS (IGNORE '-')
# --------------------------------------------
def build_consensus_ignore_dash(df, seq_cols):
    consensus = []
    for col in seq_cols:
        values = df[col][df[col] != '-']
        if len(values) == 0:
            consensus.append('-')
        else:
            consensus.append(values.mode()[0])
    return consensus


# --------------------------------------------
# EXTRACT MUTATIONS
# --------------------------------------------
def extract_mutations(seq, consensus):
    mutations = []
    for i, (aa, ref) in enumerate(zip(seq, consensus)):
        if aa != ref:
            mutations.append(f"{ref}{i+1}{aa}")
    return mutations


# --------------------------------------------
# MAIN LOOP
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
    # CLEAN ALIGNMENT COLUMNS
    # --------------------------------------------
    df[seq_cols] = df[seq_cols].astype(str)
    df[seq_cols] = df[seq_cols].apply(lambda col: col.str.strip())

    # If mixture (LM, GE, etc.) → take first residue
    df[seq_cols] = df[seq_cols].apply(lambda col: col.str[0])

    # --------------------------------------------
    # BUILD CONSENSUS (ignoring '-')
    # --------------------------------------------
    consensus_list = build_consensus_ignore_dash(df, seq_cols)

    # Replace '-' with consensus residue
    for i, col in enumerate(seq_cols):
        df.loc[df[col] == '-', col] = consensus_list[i]

    # --------------------------------------------
    # BUILD FULL SEQUENCE
    # --------------------------------------------
    df["FullSeq"] = df[seq_cols].agg("".join, axis=1)

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
        INITIAL_THRESHOLD,
        IDENTITY_SAMPLE_SIZE
    )

    print(f"Mean identity BEFORE filtering: {mean_before:.4f}")
    print(f"% pairs >= 95% BEFORE: {high_before:.2f}%")

    # --------------------------------------------
    # ADAPTIVE IDENTITY FILTERING
    # --------------------------------------------
    threshold = INITIAL_THRESHOLD
    df_final = greedy_identity_clustering(df, threshold)

    while len(df_final) < 500 and threshold < MAX_THRESHOLD:
        threshold += 0.01
        print(f"Dataset too small. Retrying with threshold = {threshold:.2f}")
        df_final = greedy_identity_clustering(df, threshold)

    print(f"Final threshold used: {threshold:.2f}")

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
        threshold,
        IDENTITY_SAMPLE_SIZE
    )

    print(f"Mean identity AFTER filtering: {mean_after:.4f}")
    print(f"% pairs >= {threshold*100:.0f}% AFTER: {high_after:.2f}%")

    # --------------------------------------------
    # MUTATION LIST
    # --------------------------------------------
    df_final["MutationList"] = df_final["FullSeq"].apply(
        lambda seq: extract_mutations(seq, consensus_list)
    )

    # --------------------------------------------
    # TARGET VALIDATION
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