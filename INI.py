import pandas as pd
import numpy as np

FILE_NAME = "INI.csv"
df = pd.read_csv(FILE_NAME)

print("\nInitial dataset size:", len(df))

sequence_cols = [col for col in df.columns if col.startswith("P") and col[1:].isdigit()]
df[sequence_cols] = df[sequence_cols].astype(str)

seq_array = df[sequence_cols].values

# Remove exact duplicates
df["FullSeq"] = df[sequence_cols].agg("".join, axis=1)
df = df.drop_duplicates(subset=["FullSeq"]).reset_index(drop=True)

seq_array = df[sequence_cols].values

print("After exact duplicate removal:", len(df))

def mutation_count(m):
    if pd.isna(m) or str(m).strip() == "":
        return 0
    return len(str(m).split(","))

df["MutCount"] = df["CompMutList"].apply(mutation_count)

original_size = len(df)
original_mean_mut = df["MutCount"].mean()

print("Original mean mutation count:", round(original_mean_mut, 2))

n = len(seq_array)
seq_length = seq_array.shape[1]

# Progressive 99% clustering with similarity metrics
threshold = 99
kept_indices = []
all_identities = []
similarity_density = []
mean_normalized_score = []

print("\nRunning FAST 99% clustering...")

for i in range(n):
    seq_i = seq_array[i]
    redundant = False
    local_similar_count = 0
    local_scores = []

    for kept_idx in kept_indices:
        seq_j = seq_array[kept_idx]
        matches = np.sum(seq_i == seq_j)
        identity = (matches / seq_length) * 100

        all_identities.append(identity)
        local_scores.append(matches)

        if identity >= threshold:
            redundant = True
            local_similar_count += 1
            break

    if not redundant:
        kept_indices.append(i)
        density = local_similar_count / n
        similarity_density.append(density)

        if local_scores:
            mean_score = np.mean(local_scores)
            normalized = mean_score / seq_length
        else:
            normalized = 1.0

        mean_normalized_score.append(normalized)

    if i % 200 == 0:
        print("Processed:", i, "/", n)

print("\nAlignment Statistics (Direct Identity):")

if all_identities:
    print("Mean Identity:", round(np.mean(all_identities), 2))
    print("Min Identity:", round(np.min(all_identities), 2))
    print("Max Identity:", round(np.max(all_identities), 2))
else:
    print("Not enough comparisons.")

df_nonredundant = df.iloc[kept_indices].reset_index(drop=True)

df_nonredundant["SimilarityDensity"] = similarity_density
df_nonredundant["MeanNormalizedScore"] = mean_normalized_score

final_size = len(df_nonredundant)
removed = original_size - final_size
mean_mut_after = df_nonredundant["MutCount"].mean()

print("\nFinal non-redundant dataset size:", final_size)
print("Sequences removed:", removed)
print("Percent removed:", round((removed / original_size) * 100, 2), "%")
print("Mean mutation count after clustering:", round(mean_mut_after, 2))

OUTPUT_FILE = FILE_NAME.replace(".csv", "_nonredundant_99percent_FAST.csv")
df_nonredundant.to_csv(OUTPUT_FILE, index=False)

print("\nFinal dataset exported as:", OUTPUT_FILE)
print("Process complete.")