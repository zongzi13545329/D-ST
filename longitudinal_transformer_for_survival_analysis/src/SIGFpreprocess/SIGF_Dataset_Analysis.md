# SIGF Dataset Analysis Report

## Dataset Overview

The SIGF (Study in the Identification of Glaucoma Features) dataset contains longitudinal fundus photography data organized for survival analysis research. The dataset is structured in three main splits: train, test, and validation.

## Dataset Versions

**Original Dataset:** `/home/lin01231/public/datasets/SIGF/` (Complete original data)
**Processed Dataset:** `/home/lin01231/public/datasets/SIGF_processed/` (Training-ready data)

## Directory Structure

### Original Dataset (SIGF)
```
/home/lin01231/public/datasets/SIGF/
├── train/
│   ├── image/all/image/
│   └── label/label/
├── test/
│   ├── image/
│   └── label/
├── validation/
│   ├── image/
│   └── label/
└── final_atten/
```

### Processed Dataset (SIGF_processed) - **RECOMMENDED FOR TRAINING**
```
/home/lin01231/public/datasets/SIGF_processed/
├── train/
│   ├── image/all/image/
│   └── label/label/
├── test/
│   ├── image/
│   └── label/
├── validation/
│   ├── image/
│   └── label/
└── final_atten/
```

## Dataset Statistics

### Original Dataset (SIGF)
- **Training set**: 2,646 images
- **Test set**: 701 images  
- **Validation set**: 337 images
- **Total**: 3,684 images

### Processed Dataset (SIGF_processed) - **FOR TRAINING USE**
- **Training set**: 2,536 images (110 progression images removed)
- **Test set**: 673 images (28 progression images removed)
- **Validation set**: 322 images (15 progression images removed)
- **Total**: 3,531 images (153 progression images removed)

**Processing Summary:**
- 37 patients with disease progression identified
- 153 post-progression images removed from training data
- Labels preserved for complete temporal sequences
- Training-ready dataset for survival analysis modeling

## Data Organization

### Image Data Structure

Images are organized differently across splits:

**Training Set Structure:**
- Images stored in patient-specific subdirectories: `/train/image/all/image/{PATIENT_EYE_ID}/`
- Format: `SD{PATIENT_ID}_{YYYY_MM_DD}_{EYE}.JPG`
- Example: `/train/image/all/image/SD1284_OS/SD1284_1990_04_11_OS.JPG`

**Test/Validation Set Structure:**
- Images stored in patient-specific subdirectories: `/test/image/{PATIENT_EYE_ID}/` or `/validation/image/{PATIENT_EYE_ID}/`
- Format: `SD{PATIENT_ID}_{YYYY_MM_DD}_{EYE}.jpg`
- Example: `/test/image/SD3434_OS/SD3434_2005_03_29_OS.jpg`

**Patient-Eye Combinations:**
- Patient ID: SD + 4-digit number
- Eye designation: OD (right eye), OS (left eye)
- Some newer images have "_01" suffix indicating potential duplicates/versions

**Example Longitudinal Series:**
- **SD1284_OS**: 7 images from 1990-1999 (progression detected at index 2)
- **SD1361_OD**: 12 images from 1990-2017 (no progression)
- **SD1459_OD**: 16 images from 1994-2017 (no progression)

### Label Data Structure

Labels are stored as text files with binary progression indicators:
- One label file per patient-eye combination
- Format: Sequential time points with 0 (no progression) or 1 (progression detected)
- Example from SD1284_OS.txt:
  ```
  0  # Time point 1: No progression
  0  # Time point 2: No progression  
  1  # Time point 3: Progression detected
  1  # Time point 4: Continued progression
  1  # Time point 5: Continued progression
  1  # Time point 6: Continued progression
  1  # Time point 7: Continued progression
  ```

## Longitudinal Data Characteristics

### Temporal Span
- **Earliest image**: 1987
- **Latest image**: 2018
- **Typical follow-up period**: 10-20 years per patient
- **Visit frequency**: Typically 1-2 years between visits

### Patient Coverage
- Each patient can contribute data from both eyes (OD and OS)
- Unbalanced follow-up periods across patients
- Some patients have very long follow-up (>25 years)

### Progression Patterns
Based on comprehensive label analysis across all splits:
- **No progression cases**: All time points labeled as 0 (majority of patients)
- **Early progression**: Transition from 0 to 1 occurs early in sequence
- **Late progression**: Transition occurs later in the timeline
- **Stable progression**: Once progression is detected (1), it typically remains stable

**Progression Statistics:**
- **Training set**: 27/301 patients (9%) have progression
- **Validation set**: 3/35 patients (8.6%) have progression  
- **Test set**: 7/67 patients (10.4%) have progression
- **Total**: 37/403 patients (9.2%) show disease progression

## Data Quality Observations

### Image Naming Conventions
- Consistent patient ID format (SD + 4-digit number)
- Date format: YYYY_MM_DD
- Eye designation: OD (right) or OS (left)
- Some newer images have "_01" suffix indicating potential duplicates/versions

### Label Consistency
- Binary labels (0/1) aligned with image time points
- Label sequence length matches number of images per patient-eye
- Some sequences show immediate progression (0→1 at early time points)
- Others show delayed progression or no progression throughout follow-up

## Technical Considerations

### For Longitudinal Modeling
1. **Variable sequence lengths**: Different patients have different numbers of visits
2. **Irregular time intervals**: Time gaps between visits vary
3. **Survival analysis ready**: Binary progression labels suitable for time-to-event modeling
4. **Multi-eye data**: Need to handle correlation between left and right eyes of same patient




## Processing Details

### Data Preprocessing
The processed dataset (SIGF_processed) has been optimized for survival analysis training:

1. **Progression Detection**: Identified 37 patients (9.2%) with disease progression
2. **Image Removal**: Removed all images from the first progression timepoint onwards
3. **Label Preservation**: Kept all original labels intact for complete temporal modeling
4. **Training Optimization**: Prevents data leakage from post-progression timepoints

### Usage Recommendations

**For Model Training:** Use `/home/lin01231/public/datasets/SIGF_processed/`
- Clean dataset without post-progression images
- Maintains temporal sequence integrity
- Optimized for survival analysis objectives

**For Data Analysis:** Reference original `/home/lin01231/public/datasets/SIGF/`
- Complete unmodified dataset
- Useful for understanding full disease progression patterns

## Dataset Suitability

This processed dataset is specifically optimized for:
- **Longitudinal survival analysis** with clean temporal sequences
- **Disease progression prediction** without data leakage
- **Time-to-event modeling** with imaging data
- **Transformer-based sequence modeling** for medical outcomes

The preprocessing ensures proper survival analysis methodology while preserving the valuable longitudinal structure needed for transformer-based models in ophthalmology.


从SIGF processed to csv files
plan A: index版本
现在我们需要把数据集做成类似/home/lin01231/zhan9191/AI4M/SongProj/longitudinal_transformer_for_survival_analysis/datasets/OHTS路径下面的train valid test csv文件
关于csv第一行内容的说明
ID：比如SD1294
LR：OD或者OS
glaucoma：都是0因为我们删除了发病后图像
year：就是同一个subject中不同图像的index，比如SD1294有3张图像，那么year就是0 1 2， time to event就是3
censorship：如果label全是0就是0，但凡有一个1代表发病后就是1
请你编写csv_index.py代码来实现我们上述转化，/home/lin01231/zhan9191/AI4M/SongProj/longitudinal_transformer_for_survival_analysis/src/SIGFpreprocess这里面写

我们现在添加一个功能：由于time to event最大为20，我们现在转化为csv文件，对于超过14张图的subject，我们跳过