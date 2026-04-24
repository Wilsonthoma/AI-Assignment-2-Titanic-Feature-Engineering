# AI Assignment 2: Titanic Survival Prediction - Feature Engineering & Selection

## Student Information
- **Name:** Wilson Thoma
- **Unit:** COMP 334 - Artificial Intelligence
- **Date:** April 2026

---

##  Assignment Overview

This project performs comprehensive data cleaning, feature engineering, and feature selection on the Titanic dataset. The goal is to transform raw passenger data into meaningful features that can predict survival outcomes.

**Dataset:** Titanic - Machine Learning from Disaster (Kaggle)
- 891 passengers in training set
- 12 original features
- Target variable: Survived (0 = No, 1 = Yes)

---

## 🔧 Part 1: Data Cleaning (10 Marks)

### Missing Values Handled

| Column | Missing % | Strategy | Value Used |
|--------|-----------|----------|------------|
| Age | 19.9% | Median imputation | 28.0 years |
| Cabin | 77.1% | Created indicator column | Cabin_Unknown = 1 if missing |
| Embarked | 0.2% | Mode imputation | 'S' (Southampton) |

### Outliers Capped

| Feature | Method | Threshold |
|---------|--------|-----------|
| Fare | 95th percentile capping | 112.08 |
| Age | 99th percentile capping | 65.87 years |

### Data Consistency
- No duplicate rows found
- Sex values standardized (male/female)
- Pclass values verified (1, 2, 3)

---

## 🛠️ Part 2: Feature Engineering (30 Marks)

### New Features Created (7 total)

| Feature | Formula | Purpose |
|---------|---------|---------|
| FamilySize | SibSp + Parch + 1 | Measure of family support |
| IsAlone | 1 if FamilySize == 1 else 0 | Identify solo travelers |
| Title | Extracted from Name | Social status indicator |
| AgeGroup | Child, Teen, Adult, Senior | Non-linear age effects |
| FarePerPerson | Fare / FamilySize | Per-person wealth |
| Fare_log | log(1 + Fare_capped) | Reduce right skewness |
| Age_log | log(1 + Age_capped) | Normalize distribution |

### Title Categories Extracted
- Mr (adult male)
- Mrs (married female)
- Miss (young/unmarried female)
- Master (young male)
- Rare (Dr, Rev, Col, Major, Lady, etc.)

### Age Group Definitions
| Group | Age Range | Survival Rate |
|-------|-----------|---------------|
| Child | 0-11 years | 60%+ |
| Teen | 12-17 years | ~50% |
| Adult | 18-59 years | ~38% |
| Senior | 60+ years | ~35% |

### Categorical Encoding

| Feature | Encoding Type |
|---------|---------------|
| Sex | One-hot encoding |
| Embarked | One-hot encoding |
| Title | One-hot encoding |
| AgeGroup | One-hot encoding |
| Pclass | Ordinal encoding (1st→3, 2nd→2, 3rd→1) |

---

##  Part 3: Feature Selection (10 Marks)

### Top 10 Features by Correlation with Survival

| Rank | Feature | Correlation | Interpretation |
|------|---------|-------------|----------------|
| 1 | Title_Mrs | +0.3420 | Married women survived most |
| 2 | Pclass_encoded | +0.3385 | Higher class = higher survival |
| 3 | Title_Miss | +0.3356 | Young women survived well |
| 4 | Fare_log | +0.3303 | Wealthy passengers survived more |
| 5 | Fare_capped | +0.3147 | Same as above |
| 6 | FarePerPerson | +0.2216 | Per-person wealth |
| 7 | AgeGroup_Child | +0.1121 | Children survived more |
| 8 | AgeGroup_Teen | +0.0498 | Teens slightly better |
| 9 | FamilySize | +0.0166 | Small families better |
| 10 | Embarked_Q | +0.0037 | Minor effect |

### Top 10 Features by Random Forest Importance

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Sex_male | 0.1224 |
| 2 | FarePerPerson | 0.1196 |
| 3 | Title_Mr | 0.1124 |
| 4 | Age_capped | 0.1051 |
| 5 | Age_log | 0.1050 |
| 6 | Fare_log | 0.0980 |
| 7 | Fare_capped | 0.0946 |
| 8 | Pclass_encoded | 0.0459 |
| 9 | FamilySize | 0.0413 |
| 10 | Title_Miss | 0.0373 |

### Final 10 Selected Features for Model

1. **Sex_male** - Gender (negative predictor)
2. **FarePerPerson** - Per-person wealth
3. **Title_Mr** - Adult male status
4. **Age_capped** - Age (capped at 99th percentile)
5. **Age_log** - Log-transformed age
6. **Fare_log** - Log-transformed fare
7. **Fare_capped** - Capped fare
8. **Pclass_encoded** - Ticket class (ordinal)
9. **FamilySize** - Family group size
10. **Title_Miss** - Young female status

---

##  Key Insights & Findings

### What Factors Most Influenced Survival?

| Factor | Finding | Evidence |
|--------|---------|----------|
| **Gender** | Women survived at ~3x higher rate | Sex_male correlation: -0.54 |
| **Social Status** | Mrs/Miss had high survival, Mr low | Title features in top 3 |
| **Class** | 1st class: 63% survival, 3rd class: 24% | Pclass_encoded correlation: +0.34 |
| **Wealth** | Higher fare = higher survival | Fare features in top 5 |
| **Family Size** | Size 2-4 had best outcomes | FamilySize correlation: +0.02 |
| **Age** | Children (<12) had 60%+ survival | AgeGroup_Child correlation: +0.11 |

### Interesting Observations

- 👩 **Women were prioritized** - "Women and children first" clearly reflected in data
- 💰 **Wealth mattered** - First class passengers had 2.6x higher survival than third class
- 👶 **Children survived more** - Especially those traveling with family
- 👤 **Alone travelers suffered** - Survival rate ~30% vs ~50% for those with family
- 🎩 **Title revealed status** - "Mr" had lowest survival, "Mrs" had highest



---

##  How to Run

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/Wilsonthoma/AI-Assignment-2-Titanic-Survival.git
cd AI-Assignment-2-Titanic-Survival

pip install -r requirements.txt
