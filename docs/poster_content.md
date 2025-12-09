# Poster Content: Predicting Poverty in India Using Transfer Learning

## TITLE
**Predicting Poverty in India from Satellite Imagery Using Transfer Learning**

Replication and Adaptation of Jean et al. (2016) | [Your Name] | UCL AI4SD 2024

---

## 1. Brief summary of the baseline method, problem, and published results

**Jean et al. (2016) "Combining satellite imagery and machine learning to predict poverty"**

**Problem:** Traditional poverty surveys are expensive and infrequent, leaving gaps in monitoring progress toward SDG 1.

**Method:** Transfer learning approach:
1. Train CNN on nightlight classification (dim/medium/bright) using 500k satellite images
2. Extract 512 features from trained CNN
3. Use Ridge regression to predict DHS wealth index

**Published Results:**
- Nightlight classification: ~75% accuracy
- Wealth prediction: R² = 0.50-0.55 across 5 African countries
- 12x improvement over nightlights-only baseline

---

## 2. Your chosen sustainable development challenge and why it matters (link to SDGs)

**Challenge: Poverty Mapping in India**

**Why India?**
- 1.4 billion people; home to 1/3 of global extreme poverty
- Significant regional inequality (urban megacities vs rural villages)
- Limited survey coverage in remote areas

**SDG Alignment:**
- **SDG 1:** No Poverty – enables targeted welfare interventions
- **SDG 10:** Reduced Inequalities – identifies spatial disparities
- **SDG 11:** Sustainable Cities – monitors urban/rural development gaps

**Policy Relevance:** Support programs like PM-KISAN, MGNREGA targeting

---

## 3. What you changed and why

| Original | Adaptation | Rationale |
|----------|------------|-----------|
| 3-band RGB | **7-band multispectral** | Capture more economic signals (nightlights, built area, terrain) |
| VGG-F (60M params) | **LightCNN256 (4M params)** | Memory efficiency; can't use ImageNet weights for 7 bands |
| ImageNet pretrained | **Train from scratch** | No pretrained models for multi-spectral input |
| 5 African countries | **India only** | Context-specific adaptation |
| 400×400 resolution | **256×256 (100m/pixel)** | Balance detail vs. computational cost |

**Same approach:** Nightlight transfer learning → Feature extraction → Ridge regression

---

## 4. Sourcing, quality and training

**Data Sources:**
- **DHS India 2019-2021:** 30,052 clusters with GPS + wealth index
- **Sentinel-2:** RGB optical (10m resolution)
- **VIIRS:** Nighttime lights (500m resolution)
- **Dynamic World:** Built area mask
- **SRTM DEM:** Elevation & slope

**Quality:**
- 28,880 valid images (96% success rate)
- Stratified split by wealth quintile (70/15/15)
- Per-band normalization (2-98 percentile)

**Training:**
- Phase 1: 17 epochs, AdamW optimizer, cosine LR schedule
- Phase 2: RidgeCV with 5-fold cross-validation

---

## 5. Quantitative results (accuracy, metrics, error analysis)

| Metric | Value |
|--------|-------|
| **Nightlight Accuracy** | 86.4% |
| **Wealth R² (Test)** | [INSERT YOUR VALUE] |
| **MAE** | [INSERT YOUR VALUE] |
| **Baseline R² (NL-only)** | ~0.03 |
| **Improvement** | [X]x over baseline |

**Error Analysis:**
- Higher errors in urban areas (complex economic patterns)
- Better performance in rural regions
- Residuals approximately normal (unbiased predictions)

---

## 6. Comparison with the results of the main paper

| Metric | Jean et al. (2016) | Our Replication |
|--------|-------------------|-----------------|
| Context | 5 African countries | India |
| Images | 500k (training) | 29k |
| NL Accuracy | ~75% | **86.4%** ✓ |
| Wealth R² | 0.50-0.55 | [YOUR R²] |
| Baseline improvement | 12x | [YOUR X]x |

**Within ±5%?** [Yes/No based on your results]

**Key difference:** Higher nightlight accuracy likely due to India's distinct urban/rural contrast

---

## 7. Contribution to sustainable development goals, ethical implications in context

**Contribution to SDGs:**
- Enables **continuous poverty monitoring** without expensive surveys
- Supports **evidence-based policy** targeting poorest regions
- Uses **freely available** satellite data (scalable globally)

**Ethical Implications:**
- **Privacy:** High-resolution imagery could enable surveillance
- **Bias:** Model trained on existing data may reinforce inequalities
- **Misuse:** Could be used to discriminate against poor areas

**Limitations:**
- Cannot capture informal economy
- Temporal lag between imagery and surveys
- Urban areas underrepresented in DHS

**Future Work:** Multi-temporal analysis, ground-truth validation

---

## FOOTER

**GitHub:** [Your Repository Link]

**References:** Jean, N., et al. (2016). Combining satellite imagery and machine learning to predict poverty. Science, 353(6301), 790-794.
