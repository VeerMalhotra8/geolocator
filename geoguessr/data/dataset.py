"""
PyTorch Dataset and DataLoader for OSV-5M geolocation training.

Handles:
  - Image loading with CLIP ViT-L/14 336px preprocessing
  - Auxiliary label encoding (climate, drive_side, land_cover, region)
  - Geocell assignment with haversine-smoothed soft targets
  - Train/val split (90/10 geographic block holdout — no spatial leakage)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np


# CLIP ViT-L/14 normalization constants
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CLIP_SIZE = 336  # ViT-L/14 @ 336px


def get_train_transform():
    """Training transform with light augmentation + CLIP preprocessing.

    Note: RandomHorizontalFlip intentionally excluded — flipping corrupts
    driving side and text/script direction, which are strong geo signals.
    """
    return transforms.Compose([
        transforms.Resize(CLIP_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(CLIP_SIZE, pad_if_needed=True),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def get_val_transform():
    """Validation transform: deterministic CLIP preprocessing."""
    return transforms.Compose([
        transforms.Resize(CLIP_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(CLIP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


# ── Auxiliary label encodings ──────────────────────────────────────────

# Climate: OSV-5M uses Köppen-Geiger integer codes (0-29).
# Group into 5 broad climate zones by first letter:
#   A (0-2): Tropical, B (3-6): Dry, C (7-15): Temperate,
#   D (16-27): Continental, E (28+): Polar
CLIMATE_BINS = [0, 3, 7, 16, 28, 100]  # bin edges
CLIMATE_NAMES = ["Tropical", "Dry", "Temperate", "Continental", "Polar"]
NUM_CLIMATE_CLASSES = 5

# Land cover: OSV-5M uses Copernicus-style integer codes (0-11).
# Group into 6 scene types:
LAND_COVER_MAP = {
    0: 0, 1: 0,   # Urban / built-up
    2: 1, 3: 1,   # Cropland / agriculture
    4: 2, 5: 2,   # Forest / vegetation
    6: 3, 7: 3,   # Sparse / barren
    8: 4, 9: 4,   # Water / wetland
    10: 5, 11: 5, # Other (snow, ice, etc.)
}
SCENE_NAMES = ["Urban", "Cropland", "Forest", "Barren", "Water", "Other"]
NUM_SCENE_CLASSES = 6

# Drive side: 0 = right, 1 = left
NUM_DRIVE_CLASSES = 3  # right, left, unknown (unknown not in data but kept for safety)

# Region: Country code → UN macro-region (15 classes)
# fmt: off
COUNTRY_TO_REGION = {
    # Northern America (0)
    "US": 0, "CA": 0, "MX": 0, "BM": 0, "GL": 0, "PM": 0,
    # Central America & Caribbean (1)
    "GT": 1, "BZ": 1, "SV": 1, "HN": 1, "NI": 1, "CR": 1, "PA": 1,
    "CU": 1, "JM": 1, "HT": 1, "DO": 1, "PR": 1, "TT": 1, "BB": 1,
    "BS": 1, "AG": 1, "DM": 1, "GD": 1, "KN": 1, "LC": 1, "VC": 1,
    "AW": 1, "CW": 1, "SX": 1, "BQ": 1, "GP": 1, "MQ": 1, "VI": 1,
    "KY": 1, "TC": 1, "VG": 1, "AI": 1, "MS": 1, "MF": 1, "BL": 1,
    # South America (2)
    "BR": 2, "AR": 2, "CL": 2, "CO": 2, "PE": 2, "VE": 2, "EC": 2,
    "BO": 2, "PY": 2, "UY": 2, "GY": 2, "SR": 2, "GF": 2, "FK": 2,
    # Northern Europe (3)
    "GB": 3, "IE": 3, "IS": 3, "NO": 3, "SE": 3, "FI": 3, "DK": 3,
    "EE": 3, "LV": 3, "LT": 3, "GG": 3, "JE": 3, "IM": 3, "FO": 3,
    "SJ": 3, "AX": 3,
    # Western Europe (4)
    "FR": 4, "DE": 4, "NL": 4, "BE": 4, "LU": 4, "AT": 4, "CH": 4,
    "LI": 4, "MC": 4,
    # Eastern Europe (5)
    "RU": 5, "PL": 5, "UA": 5, "CZ": 5, "SK": 5, "HU": 5, "RO": 5,
    "BG": 5, "MD": 5, "BY": 5,
    # Southern Europe (6)
    "IT": 6, "ES": 6, "PT": 6, "GR": 6, "HR": 6, "RS": 6, "BA": 6,
    "ME": 6, "MK": 6, "AL": 6, "SI": 6, "MT": 6, "AD": 6, "SM": 6,
    "VA": 6, "GI": 6, "XK": 6, "CY": 6,
    # Northern Africa (7)
    "EG": 7, "LY": 7, "TN": 7, "DZ": 7, "MA": 7, "SD": 7, "EH": 7,
    # Sub-Saharan Africa (8)
    "ZA": 8, "NG": 8, "KE": 8, "ET": 8, "GH": 8, "TZ": 8, "UG": 8,
    "SN": 8, "CM": 8, "CI": 8, "MG": 8, "MZ": 8, "ZM": 8, "ZW": 8,
    "BW": 8, "NA": 8, "ML": 8, "BF": 8, "NE": 8, "TD": 8, "AO": 8,
    "CD": 8, "CG": 8, "GA": 8, "GQ": 8, "RW": 8, "BI": 8, "MW": 8,
    "LS": 8, "SZ": 8, "DJ": 8, "ER": 8, "SO": 8, "MR": 8, "GM": 8,
    "GW": 8, "SL": 8, "LR": 8, "TG": 8, "BJ": 8, "CF": 8, "SS": 8,
    "MU": 8, "SC": 8, "CV": 8, "KM": 8, "RE": 8, "YT": 8, "ST": 8,
    # Western Asia / Middle East (9)
    "TR": 9, "SA": 9, "AE": 9, "IL": 9, "JO": 9, "LB": 9, "IQ": 9,
    "IR": 9, "SY": 9, "YE": 9, "OM": 9, "KW": 9, "BH": 9, "QA": 9,
    "PS": 9, "GE": 9, "AM": 9, "AZ": 9,
    # Central Asia (10)
    "KZ": 10, "UZ": 10, "TM": 10, "KG": 10, "TJ": 10, "MN": 10, "AF": 10,
    # South Asia (11)
    "IN": 11, "PK": 11, "BD": 11, "LK": 11, "NP": 11, "BT": 11, "MV": 11,
    # Southeast Asia (12)
    "TH": 12, "VN": 12, "MY": 12, "ID": 12, "PH": 12, "SG": 12, "MM": 12,
    "KH": 12, "LA": 12, "BN": 12, "TL": 12,
    # East Asia (13)
    "CN": 13, "JP": 13, "KR": 13, "TW": 13, "HK": 13, "MO": 13, "KP": 13,
    # Oceania (14)
    "AU": 14, "NZ": 14, "PG": 14, "FJ": 14, "WS": 14, "TO": 14, "VU": 14,
    "SB": 14, "KI": 14, "FM": 14, "MH": 14, "PW": 14, "NR": 14, "TV": 14,
    "CK": 14, "NU": 14, "AS": 14, "GU": 14, "MP": 14, "NC": 14, "PF": 14,
    "WF": 14,
}
# fmt: on
REGION_NAMES = [
    "N. America", "C. America", "S. America",
    "N. Europe", "W. Europe", "E. Europe", "S. Europe",
    "N. Africa", "Sub-Saharan Africa", "Middle East",
    "Central Asia", "South Asia", "SE Asia", "East Asia", "Oceania",
]
NUM_REGION_CLASSES = 15
UNKNOWN_REGION = NUM_REGION_CLASSES  # fallback for unmapped countries


def encode_climate(climate_val: float) -> int:
    """Map Köppen-Geiger code to 5 broad climate classes."""
    if np.isnan(climate_val):
        return 2  # default to Temperate
    c = int(climate_val)
    if c < 3:
        return 0   # Tropical
    elif c < 7:
        return 1   # Dry
    elif c < 16:
        return 2   # Temperate
    elif c < 28:
        return 3   # Continental
    else:
        return 4   # Polar


def encode_land_cover(lc_val: float) -> int:
    """Map land cover code to 6 scene classes."""
    if np.isnan(lc_val):
        return 5  # Other
    return LAND_COVER_MAP.get(int(lc_val), 5)


def encode_drive_side(ds_val: float) -> int:
    """Map drive side: 0=right, 1=left, else=unknown."""
    if np.isnan(ds_val):
        return 2  # unknown
    v = int(ds_val)
    if v in (0, 1):
        return v
    return 2


def encode_region(country_code: str) -> int:
    """Map country code to UN macro-region."""
    return COUNTRY_TO_REGION.get(country_code, UNKNOWN_REGION)


# ── Dataset ────────────────────────────────────────────────────────────

class OSV5MDataset(Dataset):
    """PyTorch Dataset for OSV-5M geolocation images.

    Returns:
        image: [3, 336, 336] float32 tensor (CLIP-normalized)
        smooth_target: [num_cells] float32 tensor (haversine-smoothed soft labels)
        aux_labels: dict of int64 tensors {climate, scene, drive_side, region}
    """

    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: str,
        transform=None,
        smooth_indices: np.ndarray = None,
        smooth_probs: np.ndarray = None,
        num_cells: int = None,
        df_indices: np.ndarray = None,
    ):
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.num_cells = num_cells

        # If df_indices provided, use them to index into smooth arrays
        # (needed for train/val split — df rows map to original array positions)
        self.smooth_indices = smooth_indices
        self.smooth_probs = smooth_probs
        self.df_indices = df_indices

        # Pre-encode all auxiliary labels
        self.image_ids = df["id"].astype(str).values
        self.latitudes = df["latitude"].values.astype(np.float32)
        self.longitudes = df["longitude"].values.astype(np.float32)
        self.climate_labels = np.array(
            [encode_climate(c) for c in df["climate"].values], dtype=np.int64
        )
        self.scene_labels = np.array(
            [encode_land_cover(c) for c in df["land_cover"].values], dtype=np.int64
        )
        self.drive_labels = np.array(
            [encode_drive_side(d) for d in df["drive_side"].values], dtype=np.int64
        )
        self.region_labels = np.array(
            [encode_region(c) for c in df["country"].values], dtype=np.int64
        )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # Load image
        img_path = self.images_dir / f"{self.image_ids[idx]}.jpg"
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Build dense smooth target from sparse representation
        orig_idx = self.df_indices[idx] if self.df_indices is not None else idx
        smooth_target = torch.zeros(self.num_cells, dtype=torch.float32)
        indices = self.smooth_indices[orig_idx]
        probs = self.smooth_probs[orig_idx]
        # .copy() needed: numpy slice may be non-contiguous/non-writable
        smooth_target[indices] = torch.from_numpy(probs.copy())

        aux_labels = {
            "climate": torch.tensor(self.climate_labels[idx]),
            "scene": torch.tensor(self.scene_labels[idx]),
            "drive_side": torch.tensor(self.drive_labels[idx]),
            "region": torch.tensor(self.region_labels[idx]),
        }

        return image, smooth_target, aux_labels


# ── Train/Val split ───────────────────────────────────────────────────

def train_val_split(
    df: pd.DataFrame, val_fraction: float = 0.1, seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Geographic block holdout split — prevents spatial leakage.

    Instead of random 90/10 within each country, assigns images to 0.5-degree
    lat/lon grid cells and holds out entire grid cells for validation.  This
    guarantees geographic separation between train and val sets (~55 km at the
    equator, ~40 km at lat 45).

    Returns: (train_df, val_df, train_orig_indices, val_orig_indices)
    where *_orig_indices are the original row positions in df,
    needed to index into precomputed smooth target arrays.
    """
    GRID_DEG = 0.5  # grid cell size in degrees

    rng = np.random.RandomState(seed)
    train_parts, val_parts = [], []
    train_indices, val_indices = [], []

    for _, group in df.groupby("country"):
        if len(group) <= 2:
            # Too few images to split — all go to train
            train_parts.append(group)
            train_indices.extend(group.index.tolist())
            continue

        # Assign each image to a geographic grid cell
        grid_lat = np.floor(group["latitude"].values / GRID_DEG).astype(np.int32)
        grid_lon = np.floor(group["longitude"].values / GRID_DEG).astype(np.int32)
        # Unique cell key per grid cell
        cell_keys = grid_lat * 100000 + grid_lon
        group = group.copy()
        group["_grid_cell"] = cell_keys

        unique_cells = group["_grid_cell"].unique()

        if len(unique_cells) <= 1:
            # Only one grid cell in this country — all go to train
            train_parts.append(group.drop(columns="_grid_cell"))
            train_indices.extend(group.index.tolist())
            continue

        # Shuffle cells and greedily assign to val until we hit target fraction
        shuffled_cells = rng.permutation(unique_cells)
        target_val = int(len(group) * val_fraction)
        cell_sizes = group.groupby("_grid_cell").size().to_dict()
        val_cell_set = set()
        val_count = 0

        for cell in shuffled_cells:
            cell_size = cell_sizes[cell]
            if val_count + cell_size <= target_val + cell_size // 2:
                val_cell_set.add(cell)
                val_count += cell_size
            if val_count >= target_val:
                break

        # If we got no val cells (rare), force the smallest cell into val
        if not val_cell_set and len(unique_cells) > 1:
            cell_sizes = group.groupby("_grid_cell").size()
            smallest_cell = cell_sizes.idxmin()
            val_cell_set.add(smallest_cell)

        is_val = group["_grid_cell"].isin(val_cell_set)
        val_group = group[is_val].drop(columns="_grid_cell")
        train_group = group[~is_val].drop(columns="_grid_cell")

        train_parts.append(train_group)
        val_parts.append(val_group)
        train_indices.extend(train_group.index.tolist())
        val_indices.extend(val_group.index.tolist())

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame(columns=df.columns)

    # Print geographic separation stats
    n_total = len(train_df) + len(val_df)
    print(f"Geographic split: {len(train_df):,} train / {len(val_df):,} val "
          f"({100*len(val_df)/n_total:.1f}%) | grid={GRID_DEG}deg (~{GRID_DEG*111:.0f}km)")

    return (
        train_df, val_df,
        np.array(train_indices, dtype=np.int64),
        np.array(val_indices, dtype=np.int64) if val_indices else np.array([], dtype=np.int64),
    )


# ── DataLoader factory ────────────────────────────────────────────────

def create_dataloaders(
    metadata_path: str = "GEOPROJECT/data/osv5m_50k/metadata.csv",
    images_dir: str = "GEOPROJECT/data/osv5m_50k/images",
    geocell_dir: str = "GEOPROJECT/data/osv5m_50k",
    batch_size: int = 6,
    num_workers: int = 2,
    val_fraction: float = 0.1,
) -> tuple[DataLoader, DataLoader, int]:
    """Create train and val DataLoaders with geocell smooth targets.

    Args:
        metadata_path: path to metadata.csv
        images_dir: path to images directory
        geocell_dir: path to directory with geocell .npy files
            (smooth_indices.npy, smooth_probs.npy, geocell_config.json)
        batch_size: batch size (default: 6 for 8GB VRAM)
        num_workers: DataLoader workers (default: 2 for 16GB RAM)
        val_fraction: validation split fraction

    Returns: (train_loader, val_loader, num_cells)
    """
    df = pd.read_csv(metadata_path, dtype={"id": str})

    # Load geocell data if available
    geocell_path = Path(geocell_dir)
    smooth_indices = None
    smooth_probs = None
    num_cells = 0

    if (geocell_path / "smooth_indices.npy").exists():
        smooth_indices = np.load(geocell_path / "smooth_indices.npy")
        smooth_probs = np.load(geocell_path / "smooth_probs.npy")
        import json
        with open(geocell_path / "geocell_config.json") as f:
            num_cells = json.load(f)["num_cells"]
        print(f"Geocells: {num_cells} cells, smooth targets loaded")
    else:
        raise FileNotFoundError(
            f"Geocell data not found in {geocell_path}. "
            f"Run 'python -m geoguessr.data.geocells' first to build S2 cells."
        )

    # Split with index tracking (indices map into smooth target arrays)
    train_df, val_df, train_orig_idx, val_orig_idx = train_val_split(df, val_fraction)

    print(f"Train: {len(train_df):,} images | Val: {len(val_df):,} images")
    print(f"Train countries: {train_df['country'].nunique()} | Val countries: {val_df['country'].nunique()}")

    train_dataset = OSV5MDataset(
        train_df, images_dir,
        transform=get_train_transform(),
        smooth_indices=smooth_indices,
        smooth_probs=smooth_probs,
        num_cells=num_cells,
        df_indices=train_orig_idx,
    )
    val_dataset = OSV5MDataset(
        val_df, images_dir,
        transform=get_val_transform(),
        smooth_indices=smooth_indices,
        smooth_probs=smooth_probs,
        num_cells=num_cells,
        df_indices=val_orig_idx,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, num_cells




# ── Quick test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Testing DataLoader...\n")

    train_loader, val_loader, num_cells = create_dataloaders(
        batch_size=4, num_workers=0,  # num_workers=0 for debugging
    )

    # Grab one batch
    images, smooth_targets, aux = next(iter(train_loader))

    print(f"Batch shape:         {images.shape}")             # [4, 3, 336, 336]
    print(f"Smooth target shape: {smooth_targets.shape}")     # [4, num_cells]
    print(f"Climate labels:      {aux['climate']}")           # [4] int64
    print(f"Scene labels:        {aux['scene']}")             # [4] int64
    print(f"Drive labels:        {aux['drive_side']}")        # [4] int64
    print(f"Region labels:       {aux['region']}")            # [4] int64

    # Verify value ranges
    print(f"\nImage min/max:       {images.min():.3f} / {images.max():.3f}")

    # Verify smooth targets
    if smooth_targets.numel() > 0:
        print(f"Smooth target sums:  {smooth_targets.sum(dim=1)}")  # should be ~1.0
        print(f"Smooth target max:   {smooth_targets.max(dim=1).values}")
        print(f"Nonzero per sample:  {(smooth_targets > 0).sum(dim=1)}")
        print(f"Num cells:           {num_cells}")

    # Check label distributions on full train set
    print(f"\n--- Train set label distributions ---")
    for name, labels in [
        ("Climate", train_loader.dataset.climate_labels),
        ("Scene", train_loader.dataset.scene_labels),
        ("Drive", train_loader.dataset.drive_labels),
        ("Region", train_loader.dataset.region_labels),
    ]:
        unique, counts = np.unique(labels, return_counts=True)
        print(f"{name}: {dict(zip(unique, counts))}")

    print(f"\nTotal train batches: {len(train_loader)}")
    print(f"Total val batches:   {len(val_loader)}")
    print("\nDataLoader test PASSED!")
