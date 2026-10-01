# Generator Generalization Leakage Audit

## Generator: audioldm2
- **[PASS]** Test ytids are completely disjoint from training ytids
- **[PASS]** No test ytid appears in training through another AI generator
- **[PASS]** No test ytid appears in human training set
- **[PASS]** Human clips paired with test ytids excluded from training
- **[PASS]** Split is exactly 80 train / 20 test ytids

- **Unique Train ytids:** 80
- **Unique Test ytids:** 20
- **Train/Test Intersection:** 0
- **[PASS]** Test set only contains 'human' and 'audioldm2'
- **[PASS]** Train set does NOT contain 'audioldm2'

**Model LR reproducibility:**
- **[PASS]** F1 metric documented
- **[PASS]** ROC-AUC metric documented

**Model RF reproducibility:**
- **[PASS]** F1 metric documented
- **[PASS]** ROC-AUC metric documented


## Generator: stable_audio_open
- **[PASS]** Test ytids are completely disjoint from training ytids
- **[PASS]** No test ytid appears in training through another AI generator
- **[PASS]** No test ytid appears in human training set
- **[PASS]** Human clips paired with test ytids excluded from training
- **[PASS]** Split is exactly 80 train / 20 test ytids

- **Unique Train ytids:** 80
- **Unique Test ytids:** 20
- **Train/Test Intersection:** 0
- **[PASS]** Test set only contains 'human' and 'stable_audio_open'
- **[PASS]** Train set does NOT contain 'stable_audio_open'

**Model LR reproducibility:**
- **[PASS]** F1 metric documented
- **[PASS]** ROC-AUC metric documented

**Model RF reproducibility:**
- **[PASS]** F1 metric documented
- **[PASS]** ROC-AUC metric documented


## Generator: musicldm
- **[PASS]** Test ytids are completely disjoint from training ytids
- **[PASS]** No test ytid appears in training through another AI generator
- **[PASS]** No test ytid appears in human training set
- **[PASS]** Human clips paired with test ytids excluded from training
- **[PASS]** Split is exactly 80 train / 20 test ytids

- **Unique Train ytids:** 80
- **Unique Test ytids:** 20
- **Train/Test Intersection:** 0
- **[PASS]** Test set only contains 'human' and 'musicldm'
- **[PASS]** Train set does NOT contain 'musicldm'

**Model LR reproducibility:**
- **[PASS]** F1 metric documented
- **[PASS]** ROC-AUC metric documented

**Model RF reproducibility:**
- **[PASS]** F1 metric documented
- **[PASS]** ROC-AUC metric documented


## Generator: mustango
- **[PASS]** Test ytids are completely disjoint from training ytids
- **[PASS]** No test ytid appears in training through another AI generator
- **[PASS]** No test ytid appears in human training set
- **[PASS]** Human clips paired with test ytids excluded from training
- **[PASS]** Split is exactly 80 train / 20 test ytids

- **Unique Train ytids:** 80
- **Unique Test ytids:** 20
- **Train/Test Intersection:** 0
- **[PASS]** Test set only contains 'human' and 'mustango'
- **[PASS]** Train set does NOT contain 'mustango'

**Model LR reproducibility:**
- **[PASS]** F1 metric documented
- **[PASS]** ROC-AUC metric documented

**Model RF reproducibility:**
- **[PASS]** F1 metric documented
- **[PASS]** ROC-AUC metric documented


## Generator: MusicGen_medium
- **[PASS]** Test ytids are completely disjoint from training ytids
- **[PASS]** No test ytid appears in training through another AI generator
- **[PASS]** No test ytid appears in human training set
- **[PASS]** Human clips paired with test ytids excluded from training
- **[PASS]** Split is exactly 80 train / 20 test ytids

- **Unique Train ytids:** 80
- **Unique Test ytids:** 20
- **Train/Test Intersection:** 0
- **[PASS]** Test set only contains 'human' and 'MusicGen_medium'
- **[PASS]** Train set does NOT contain 'MusicGen_medium'

**Model LR reproducibility:**
- **[PASS]** F1 metric documented
- **[PASS]** ROC-AUC metric documented

**Model RF reproducibility:**
- **[PASS]** F1 metric documented
- **[PASS]** ROC-AUC metric documented


## General Checks
- **[PASS]** No test features were used for model fitting, scaling, feature selection, or hyperparameter selection. `ModelTrainer` uses `sklearn.pipeline.Pipeline` which strictly fits the `StandardScaler` only on the training set fold.
- **[PASS]** Generators are exactly as specified

### Final Audit Status: PASS