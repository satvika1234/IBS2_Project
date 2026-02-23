import torch












































































import esm
import pandas as pd
import numpy as np
from tqdm import tqdm

# =====================================================
# REFERENCES
# =====================================================
PROTEASE_REF = (
    "PQVTLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"
)

RT_REF = (
    "PISPIETVPVKLKPGMDGPKVKQWPLTEEKIKALVEICTEMEKEGKISKIGPENPYNTPVFAIKKKDSTKWRKLVDFRELNKRTQDFWEVQLGIPHPAGL"
    "KKKKSVTVLDVGDAYFSVPLDEDFRKYTAFTIPSINNETPGIRYQYNVLPQGWKGSPAIFQSSMTKILEPFRKQNPDIVIYQYMDDLYVGSDLEIGQHRT"
    "KIEELRQHLLRWGLTTPDKKHQKEPPFLWMGYELHPDKWTVQPIVLPEKDSWTVNDIQKLVGKLNWASQIYPGIKVRQLCKLLRGTKALTEVIPLTEEAE"
    "LELAENREILKEPVHGVYYDPSKDLIAEIQKQGQGQWTYQIYQEPFKNLKTGKYARMRGAHTNDVKQLTEAVQKITTESIVIWGKTPKFKLPIQKETWET"
    "WWTEYWQATWIPEWEFVNTPPLVKLWYQLEKEPIVGAETFYVDGAANRETKLGKAGYVTNRGRQKVVTLTDTTNQKTELQAIYLALQDSGLEVNIVTDSQ"
    "YALGIIQAQPDQSESELVNQIIEQLIKKEKVYLAWVPAHKGIGGNEQVDKLVSAGIRKVL"
)

INI_REF = (
    "FLDGIDKAQDEHEKYHSNWRAMASDFNLPPVVAKEIVASCDKCQLKGEAMHGQVDCSPGIWQLDCTHLEGKVILVAVHVASGYIEAEVIPAETGQETAYFLL"
    "KLAGRWPVKTIHTDNGSNFTGATVRAACWWAGIKQEFGIPYNPQSQGVVESMNKELKKIIGQVRDQAEHLKTAVQMAVFIHNFKRKGGIGGYSAGERIVDI"
    "IATDIQTKELQKQITKIQNFRVYYRDSRNPLWKGPAKLLWKGEGAVVIQDNSDIKVVPRRKAKIIRDYGKQMAGDDCVASRQDED"
)

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Characters that should be treated as "same as reference"
REF_INDICATORS = {".", "-", "~", "", "nan", "None", "X", "x", "*"}

DATASETS = {
    "PI_FINAL.csv":   PROTEASE_REF,
    "NRTI_FINAL.csv": RT_REF,
    "NNRTI_FINAL.csv": RT_REF,
    "INI_FINAL.csv":  INI_REF,
}

BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# =====================================================
# LOAD ESM-2 ONCE
# =====================================================
print("Loading ESM-2 model...")
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.to(DEVICE)
model.eval()
batch_converter = alphabet.get_batch_converter()

# =====================================================
# FUNCTION: DIAGNOSE DATASET
# =====================================================
def diagnose_dataset(df, seq_cols, reference, file_name):
    print(f"\n--- Diagnostics for {file_name} ---")

    # Check a few raw values to understand format
    print("Sample raw values from first row:")
    for col in seq_cols[:10]:
        print(f"  {col}: '{df[col].iloc[0]}'")

    # Find all unique characters across all position columns
    all_chars = set()
    for col in seq_cols:
        all_chars.update(df[col].astype(str).str.strip().unique())
    print(f"All unique values in position columns: {sorted(all_chars)}")

    invalid_chars = {c for c in all_chars if c not in VALID_AA and c not in REF_INDICATORS}
    print(f"Characters not in VALID_AA and not reference indicators: {sorted(invalid_chars)}")

# =====================================================
# FUNCTION: RECONSTRUCT SEQUENCE
# =====================================================
def reconstruct_sequence(row, seq_cols, reference, extra_ref_chars=None):
    if extra_ref_chars is None:
        extra_ref_chars = set()
    ref_chars = REF_INDICATORS | extra_ref_chars
    seq = []
    for i, col in enumerate(seq_cols):
        value = str(row[col]).strip()
        if value in ref_chars or len(value) != 1:
            # Use reference if value is a placeholder or not a single character
            seq.append(reference[i])
        else:
            seq.append(value)
    return "".join(seq)

# =====================================================
# FUNCTION: EXTRACT EMBEDDINGS
# =====================================================
def extract_embeddings(seqs):
    if len(seqs) == 0:
        return np.array([])

    embeddings = []
    for i in tqdm(range(0, len(seqs), BATCH_SIZE)):
        batch_seqs = seqs[i:i + BATCH_SIZE]
        batch_data = [(str(j), seq) for j, seq in enumerate(batch_seqs)]
        labels, strs, tokens = batch_converter(batch_data)
        tokens = tokens.to(DEVICE)
        with torch.no_grad():
            results = model(tokens, repr_layers=[33])
        reps = results["representations"][33]
        for j, seq in enumerate(batch_seqs):
            seq_len = len(seq)
            embedding = reps[j, 1:seq_len + 1].mean(0)
            embeddings.append(embedding.cpu().numpy())
    return np.array(embeddings)

# =====================================================
# MAIN LOOP
# =====================================================
for file_name, reference in DATASETS.items():
    print("\n========================================")
    print("Processing:", file_name)

    df = pd.read_csv(file_name)
    print("Initial size:", len(df))

    # Identify position columns (e.g. P1, P2, ... or just numeric-named)
    seq_cols = [c for c in df.columns if c.startswith("P") and c[1:].isdigit()]
    seq_cols = sorted(seq_cols, key=lambda x: int(x[1:]))

    print("Alignment length (columns):", len(seq_cols))
    print("Reference length:", len(reference))

    if len(seq_cols) != len(reference):
        print("WARNING: Reference length mismatch. Skipping.")
        continue

    # Run diagnostics to understand your data format
    diagnose_dataset(df, seq_cols, reference, file_name)

    # Reconstruct full sequences
    df["FullSeq"] = df.apply(
        lambda row: reconstruct_sequence(row, seq_cols, reference),
        axis=1
    )

    # Show any remaining invalid characters after reconstruction
    all_reconstructed_chars = set("".join(df["FullSeq"].tolist()))
    still_invalid = all_reconstructed_chars - VALID_AA
    if still_invalid:
        print(f"Characters still invalid after reconstruction: {sorted(still_invalid)}")
        print("These rows will be removed. Consider adding them to REF_INDICATORS if they mean 'same as reference'.")

    # Filter out sequences with invalid amino acids
    valid_mask = df["FullSeq"].apply(lambda s: all(r in VALID_AA for r in s))
    removed = (~valid_mask).sum()
    df = df[valid_mask].reset_index(drop=True)
    print(f"Removed invalid sequences: {removed}")
    print(f"Remaining: {len(df)}")

    if len(df) == 0:
        print("No valid sequences remain. Skipping embedding extraction.")
        print("Check the diagnostics above — add unexpected characters to REF_INDICATORS if they mean 'same as reference'.")
        continue

    sequences = df["FullSeq"].tolist()

    print("Extracting embeddings...")
    embeddings = extract_embeddings(sequences)
    print("Embedding shape:", embeddings.shape)

    # Save embeddings
    output_file = file_name.replace(".csv", "_ESM2.npy")
    np.save(output_file, embeddings)
    print("Saved:", output_file)

    # Also save the filtered dataframe (without FullSeq) for reference
    meta_file = file_name.replace(".csv", "_filtered.csv")
    df.drop(columns=["FullSeq"]).to_csv(meta_file, index=False)
    print("Saved filtered metadata:", meta_file)

    # Load back and verify
    print("Verifying saved file...")
    loaded = np.load(output_file)
    print("Loaded shape:", loaded.shape)
    readable_df = pd.DataFrame(loaded)
    print("First 3 rows (first 5 dims shown):")
    print(readable_df.iloc[:3, :5])

print("\nAll datasets processed.")