# SIGF Dataset Integration Plan

## Overview
This document outlines the plan to integrate the SIGF (Study in the Identification of Glaucoma Features) dataset into the longitudinal transformer for survival analysis framework, following the patterns established by AREDS and OHTS datasets.

## Current Dataset Integration Analysis

### AREDS Dataset Structure
- **CSV Format**: `072123_{split}.csv` 
- **Key Fields**:
  - `RC_ID`: Patient identifier (string)
  - `LATERALITY`: Eye side
  - `IMG`: Image filename 
  - `YEAR`: Time in years
  - `time_to_event`: Event time
  - `censorship`: Censorship status
  - `AMDSEV`: AMD severity score (used for prior visit encoding)
- **Image Path**: `{data_dir}/{RC_ID}/{IMG}`
- **Label Processing**: `time_to_event * 2` (converts to 6-month bins)
- **Classes**: 27 (6-month intervals for 13 years)
- **Normalization**: `mean=(0.39293602, 0.21741964, 0.12034029), std=(0.3375143, 0.19220693, 0.10813437)`

### OHTS Dataset Structure  
- **CSV Format**: `072823_{split}.csv`
- **Key Fields**:
  - `ran`: Patient identifier (numeric)
  - `lr`: Eye laterality
  - `tracking`: Tracking ID for image filename
  - `year`: Time index
  - `time_to_event`: Event time
  - `censorship`: Censorship status
- **Image Path**: `{data_dir}/{ran}/{ran}-{tracking}.jpg`
- **Label Processing**: `time_to_event.astype(int)` (yearly bins)
- **Classes**: 15 (1-year intervals for 14 years)
- **Normalization**: `mean=(0.52438926, 0.34920336, 0.21347666), std=(0.19437555, 0.12752805, 0.06420696)`

### SIGF Dataset Structure (Generated)
- **CSV Format**: `{split}.csv` (train.csv, test.csv, validation.csv)
- **Key Fields**:
  - `ID`: Patient identifier (e.g., SD1284)
  - `LR`: Eye laterality (OD/OS)
  - `glaucoma`: Always 0 (post-progression images removed)
  - `year`: Index-based time (0, 1, 2, ...)
  - `time_to_event`: Total number of images for patient-eye
  - `censorship`: 0 (no progression) or 1 (progression occurred)
- **Image Path**: Need to determine structure
- **Label Processing**: Use `year` directly as time index
- **Classes**: **21 classes** (max `time_to_event` = 20, so need 0-20 = 21 classes)
- **Time_to_event Distribution**: 
  - Range: 6-20 images per patient-eye
  - Max values: train=20, test=20, validation=18
  - Unique values: [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20]

## SIGF Preprocessing Analysis from DeepGF

### DeepGF SIGF Processing Pipeline
Based on analysis of `/home/lin01231/zhan9191/AI4M/SongProj/DeepGF/data_processing.py`:

**Image Processing**:
- **Resize**: All fundus images resized to `(224, 224)` pixels
- **Normalization**: Simple division by 255.0 (converts 0-255 to 0-1 range)
- **Format**: RGB images, 8-bit depth, JPG format
- **No ImageNet-style normalization**: No mean/std statistics used

**Multi-Modal Structure**:
- **Fundus images**: 224×224×3 (RGB)
- **Attention maps**: 112×112×1 (grayscale, resized)
- **Polar maps**: 224×224×3 (RGB)

**Data Augmentation**:
- **Limited augmentation**: No geometric/photometric transformations found
- **Data balancing**: Positive samples oversampled 20× for class balance
- **Sequential structure**: 5 images per patient sequence

**Temporal Handling**:
- **Fixed sequence length**: 5 time points per patient
- **Year differences**: Computed between sequential visits
- **Multi-time labeling**: 5 labels per sequence

## SIGF Integration Plan

### 1. Image Path Structure Analysis
Based on SIGF_processed structure:
```
/home/lin01231/public/datasets/SIGF_processed/
├── train/image/all/image/{PATIENT_EYE_ID}/
├── test/image/{PATIENT_EYE_ID}/
└── validation/image/{PATIENT_EYE_ID}/
```

**Proposed Image Path Strategy**:
- **Training**: `{data_dir}/train/image/all/image/{ID}_{LR}/{filename}`
- **Test/Val**: `{data_dir}/{split}/image/{ID}_{LR}/{filename}`
- **Filename Pattern**: `{ID}_{YYYY_MM_DD}_{LR}.JPG` or similar

### 2. Dataset Classes Implementation

#### 2.1 SIGF_Survival_Dataset (Single Image)
```python
class SIGF_Survival_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_dir, split, augment, tpe_mode, learned_pe):
        # Similar structure to OHTS_Survival_Dataset
        self.label_df = pd.read_csv(os.path.join(self.label_dir, f'{split}.csv'))
        # Convert ID from 'SD1284' format if needed
        self.label_df['label'] = self.label_df['year'].astype(int)  # Use year directly
        
        # SIGF preprocessing - EXACT match to DeepGF approach
        # NO data augmentation (DeepGF doesn't use any geometric/photometric augmentation)
        # Only resize and simple 0-1 normalization
        self.transform = A.Compose([
            A.Resize(224, 224),  # Match DeepGF: image.resize((224, 224))
            # DeepGF normalization: image = image / 255.0
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
            albumentations.pytorch.ToTensorV2(p=1)
        ])
        # Note: augment parameter ignored to maintain DeepGF compatibility
```

#### 2.2 SIGF_Longitudinal_Survival_Dataset
```python
class SIGF_Longitudinal_Survival_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_dir, split, augment, tpe_mode, learned_pe):
        # Group by patient-eye combinations
        self.eye_dfs = self.label_df.groupby(['ID', 'LR'])
        self.eye_ids = [i for i, _ in self.eye_dfs]
        
        # Set maximum sequence length (based on analysis - SIGF has variable lengths)
        self.max_seq_length = 14  # Match AREDS/OHTS framework requirement
        
        # SIGF preprocessing - EXACTLY match DeepGF approach
        # NO augmentation regardless of augment parameter to maintain compatibility
        self.transform = A.Compose([
            A.Resize(224, 224),  # DeepGF: image.resize((224, 224))
            # DeepGF: image = image / 255.0 (simple 0-1 normalization)
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
            albumentations.pytorch.ToTensorV2(p=1)
        ])
        # Note: DeepGF uses no geometric/photometric augmentation, only oversampling
```

### 3. Field Mapping Strategy

| SIGF Field | AREDS Equivalent | OHTS Equivalent | Usage |
|------------|------------------|-----------------|-------|
| `ID` | `RC_ID` | `ran` | Patient identifier |
| `LR` | `LATERALITY` | `lr` | Eye laterality |
| `year` | `YEAR` | `year` | Time index (already index-based) |
| `time_to_event` | `time_to_event` | `time_to_event` | Event time |
| `censorship` | `censorship` | `censorship` | Censorship status |
| `glaucoma` | N/A | N/A | Disease status (always 0) |

### 4. Key Implementation Considerations

#### 4.1 Time Encoding
- **Current**: SIGF uses index-based `year` (0, 1, 2, ...)
- **TPE Modes**:
  - `bins`: Use `year` directly
  - `months`: Convert to months if actual visit dates available
- **Learned PE**: Use `year` values directly

#### 4.2 Image Loading Strategy
Need to handle different directory structures across splits:
```python
def _get_image_path(self, patient_id, laterality, split):
    patient_eye_id = f"{patient_id}_{laterality}"
    if split == 'train':
        return os.path.join(self.data_dir, 'train', 'image', 'all', 'image', patient_eye_id)
    else:
        return os.path.join(self.data_dir, split, 'image', patient_eye_id)
```

#### 4.3 Missing Features
- **No AMD Severity**: Set `prior_AMD_sev` to None (like OHTS)
- **Filename Reconstruction**: Need to reconstruct actual filenames from patient data

### 5. Collate Functions

#### 5.1 SIGF_collate_fn (for LTSA)
```python
def SIGF_collate_fn(data):
    # Similar to OHTS_collate_fn
    # Handle None values for prior_AMD_sev
    return {
        'x': x, 'y': y, 'censorship': censorship,
        'rel_time': rel_time, 'prior_AMD_sev': None,
        # ... other fields
    }
```

#### 5.2 SIGF_collate_fn_sf (for SF model)
```python
def SIGF_collate_fn_sf(data):
    # Similar to OHTS_collate_fn_sf
    # Maintain temporal structure (B, T, C, H, W)
```

### 6. Train.py Integration

Add SIGF option to main function:
```python
elif args.dataset == 'SIGF':
    args.data_dir = '/home/lin01231/public/datasets/SIGF_processed'
    args.label_dir = '/home/lin01231/zhan9191/AI4M/SongProj/longitudinal_transformer_for_survival_analysis/datasets/SIGF_processed'
    args.n_classes = 21  # Max time_to_event = 20, so need 0-20 = 21 classes
    
    if args.model == 'LTSA':
        dataset = SIGF_Longitudinal_Survival_Dataset
        collate_fn = SIGF_collate_fn
    elif args.model == 'SF':
        dataset = SIGF_Longitudinal_Survival_Dataset
        collate_fn = SIGF_collate_fn_sf
    # ... other model types
```

### 7. Normalization Parameters (DeepGF-Compatible)

SIGF uses ONLY simple 0-1 normalization to maintain DeepGF compatibility:
```python
# EXACT DeepGF normalization (from DeepGF/data_processing.py)
# DeepGF code: image = image / 255.0
# This converts 0-255 range to 0-1 range with NO mean/std normalization

# REQUIRED for albumentations compatibility:
SIGF_MEAN = (0.0, 0.0, 0.0)  # No mean subtraction
SIGF_STD = (1.0, 1.0, 1.0)   # No std division  
# max_pixel_value=255.0 ensures proper 0-1 scaling

# DO NOT USE ImageNet-style normalization - this would break DeepGF compatibility
```

### 8. Implementation Steps (DeepGF-Aligned)

1. **Create Dataset Classes**:
   - `SIGF_Survival_Dataset` with EXACT DeepGF preprocessing (224×224, 0-1 norm, no augment)
   - `SIGF_Longitudinal_Survival_Dataset` with same preprocessing approach
   - `VFM_SIGF_*` variants (may need different normalization for VFM compatibility)

2. **Implement Collate Functions**:
   - `SIGF_collate_fn` (handle None for prior_AMD_sev like OHTS)
   - `SIGF_collate_fn_sf` (maintain temporal structure)

3. **Add to train.py**:
   - Dataset selection logic with SIGF option
   - Data/label directory paths for SIGF_processed
   - Number of classes configuration: **args.n_classes = 21** (max time_to_event = 20)

4. **Preprocessing Requirements (NON-NEGOTIABLE)**:
   - ✅ 224×224 resize (matches DeepGF)
   - ✅ Simple 0-1 normalization (pixel/255.0)
   - ❌ NO data augmentation (DeepGF doesn't use any)
   - ❌ NO ImageNet-style mean/std normalization

5. **Handle Image Loading**:
   - Implement filename reconstruction from CSV data
   - Handle different directory structures per split
   - Add robust error handling for missing images
   - Ensure 224×224 output size for framework compatibility

6. **Framework Compatibility Checks**:
   - Verify tensor shapes: (B, C, H, W) = (batch, 3, 224, 224)
   - Ensure pixel values in [0, 1] range
   - Test with existing model architectures (LTSA, SF)
   - Validate collate function outputs

7. **Testing & Validation**:
   - Test single image loading matches DeepGF preprocessing
   - Test longitudinal sequence formation
   - Verify no augmentation is applied
   - Compare pixel value distributions with DeepGF output
   - Test integration with existing models

### 9. Potential Challenges

1. **Directory Structure Inconsistency**: Different structures for train vs test/val
2. **Filename Reconstruction**: Need to map from CSV data to actual filenames
3. **Missing Progression Cases**: All censorship might be 0 due to preprocessing
4. **Time Encoding**: Index-based vs actual temporal relationships
5. **Missing Clinical Data**: No equivalent to AMD severity scores
6. **Normalization Differences**: 
   - DeepGF uses simple 0-1 normalization vs ImageNet-style for AREDS/OHTS
   - May need to choose between compatibility approaches
7. **Limited Original Augmentation**: 
   - DeepGF relies mainly on oversampling, not geometric/photometric augmentation
   - Adding augmentation may improve performance but changes preprocessing
8. **Multi-Modal Architecture**: 
   - Original SIGF uses 3 streams (fundus, attention, polar)
   - Current framework uses single-stream architecture
9. **Sequence Length Variability**:
   - SIGF has variable sequence lengths (not fixed like DeepGF's 5-image sequences)
   - Need proper padding and masking strategies

### 10. Success Criteria

- [ ] SIGF datasets load without errors
- [ ] Longitudinal sequences are properly formed
- [ ] Collate functions produce correct tensor shapes
- [ ] Models can train on SIGF data
- [ ] Evaluation metrics are computed correctly
- [ ] Memory usage is reasonable for large sequences

### 11. Final Preprocessing Strategy (REQUIREMENT)

Based on your requirements to align with DeepGF preprocessing:

#### **MANDATORY Approach: DeepGF-Exact Compatibility**
- ✅ **Use simple 0-1 normalization ONLY** (`pixel/255.0`)
- ✅ **NO mean/std normalization** (no ImageNet-style stats)
- ✅ **NO data augmentation** (DeepGF doesn't use any geometric/photometric transforms)
- ✅ **224×224 resizing** (matches DeepGF exactly)
- ✅ **RGB format** (3 channels)

#### **Implementation Details**:
```python
# EXACT DeepGF preprocessing pipeline
transform = A.Compose([
    A.Resize(224, 224),  # DeepGF: image.resize((224, 224))
    A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
    albumentations.pytorch.ToTensorV2(p=1)
])
# This produces tensors with shape (3, 224, 224) and values in [0, 1]
```

#### **Framework Compatibility Requirements**:
- ✅ **Tensor shape**: (B, 3, 224, 224) for batch processing
- ✅ **Pixel range**: [0, 1] (compatible with model expectations)
- ✅ **Data type**: torch.float32
- ✅ **Sequence handling**: Variable length with padding to max_seq_length=14

**This approach ensures 100% DeepGF preprocessing compatibility while maintaining framework integration.**

### 12. Code Implementation Priority (DeepGF-Aligned)

1. **Critical Priority**: Basic dataset loading with EXACT DeepGF preprocessing
   - 224×224 resize only
   - Simple 0-1 normalization only
   - No augmentation
   
2. **High Priority**: Framework integration compatibility
   - Proper tensor shapes and data types
   - Collate function implementation
   - train.py integration

3. **Medium Priority**: Sequence handling and padding
   - Variable length sequence support
   - Proper masking for longitudinal models

4. **Future Consideration**: VFM integration
   - May require different normalization approach
   - Separate VFM_SIGF classes if needed

### 13. Success Criteria (Updated)

- [ ] ✅ SIGF preprocessing EXACTLY matches DeepGF (224×224, 0-1 norm, no augment)
- [ ] ✅ Dataset loads without errors with correct tensor shapes
- [ ] ✅ Longitudinal sequences properly formed and padded
- [ ] ✅ Collate functions produce correct tensor shapes for models
- [ ] ✅ Models can train on SIGF data without preprocessing conflicts
- [ ] ✅ Pixel value ranges [0, 1] maintained throughout pipeline
- [ ] ✅ No data augmentation applied (maintaining DeepGF fidelity)

This implementation ensures 100% DeepGF preprocessing compatibility while successfully integrating SIGF into the longitudinal transformer framework.