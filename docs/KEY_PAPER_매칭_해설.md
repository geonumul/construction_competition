# KEY PAPER ↔ 본 연구 섹션별 매칭 해설

본 문서는 KEY PAPER(Qurat Ul Ain & Rather, 2025)의 각 섹션을 본 연구(건설업 산업재해)에 어떻게 대응·변환하였는지를 정리한다.

---

## Methodology

### KEY PAPER 원문

> The methodology for this study is structured into three main components: data pre-processing, statistical modeling using machine learning, and interpretation of results with SHAP values.

### 본 연구 적용

본 연구의 방법론은 **데이터 전처리, 통계적 모델링과 머신러닝, 그리고 SHAP을 통한 결과 해석**의 세 가지 주요 구성요소로 구조화된다. 특히 본 연구는 **외부 기관의 조절효과를 검증하기 위해 독립변수를 A 그룹(내부 안전관리)과 B 그룹(현장 안전 행동)으로 분리하여 계층적으로 분석**하였다. 아래에서는 분석 전반에 걸쳐 적용된 세부 단계, 기법, 수식, 알고리즘을 개괄한다.

---

## 3.1 제외 기준 (Exclusion Criteria)

### KEY PAPER 원문

> This is a retrospective, cross-sectional study. Inclusion criteria were complete clinical records of patients aged 18–80 years with relevant lifestyle and physiological data. Exclusion criteria are detailed below. The exclusion criteria for this study were designed to ensure data quality, cohort homogeneity, and representativeness of the target population. Patients with incomplete or inconsistent data, such as missing values in key parameters (Age, BMI, Blood pressure, Glucose levels) or duplicated entries, were excluded during the preprocessing stage, accounting for the removal of 203 patients. An additional 97 patients were excluded based on study-specific criteria, including age restrictions (patients under 18 or over 80 years), the presence of terminal illnesses or unrelated chronic conditions (Advanced cancer, Severe kidney failure), and unreliable reports of lifestyle factors such as smoking or alcohol consumption. Patients who had undergone major surgeries or clinical interventions in the past six months were also excluded to avoid transient effects. Finally, outliers in essential variables (Implausibly high glucose levels without confirmed diabetes) and inconsistent parameter combinations (Extremely high BMI with normal blood pressure) were removed. These criteria ensured a robust dataset of 1200 patients suitable for statistical modeling and analysis, followed by a flowchart in (Fig. 1) illustrating the patient selection process.
>
> The dataset used in this study was synthetically generated to simulate a realistic cross-sectional clinical cohort comprising 1200 adult patients aged 18–80 years.

### 본 연구 적용

본 연구는 후향적, 횡단면 연구이다. 포함 기준은 한국산업안전보건공단의 제10차 산업안전보건 실태조사(2021)에 응답한 건설업 사업장 중 종속변수(사고발생) 및 주요 독립변수의 응답이 완전한 사례로 한정하였다. 제외 기준은 데이터 품질, 코호트 동질성, 대상 모집단에 대한 대표성을 확보하기 위해 설계되었다.

1. 종속변수(사고발생) 판단이 불가한 사례 16건과 입력 오류로 의심되는 이상치 1건(50~120억 규모·종사자 10명 현장에서 사망 30건 기록)이 우선 제외되었다.
2. 이후 주요 독립변수의 무응답 사례가 단계적으로 제외되었다:
   - 안전조직 전담부서 무응답(Q6=9) 21건
   - 위원회 운영 무응답(Q10=4 또는 9) 62건
   - 위험성평가 구조적 미응답(Q14=NaN) 24건
   - 전문지도 수혜 무응답(Q9=9) 3건

이러한 완전사례분석(Listwise Deletion) 기준은 박천수(2023, 2024, 2025)의 동일 데이터 선행연구에서 채택된 방식과 동일하다.

**본 데이터는 결측의 성격이 의료 측정의 우연 누락이 아니라 설문 응답 거부·자격 미달(Skip Logic)이므로, KNN 등 imputation 기법보다 listwise deletion이 적절하다. 무응답 자체가 정보를 담고 있어 인위적 대체는 편향을 유발할 수 있기 때문이다.**

최종적으로 1,502개 사업장 중 127개가 제외되어 **1,375개 사업장**이 분석에 사용되었다.

> - 코호트: 같은 시기에 태어났거나, 같은 시대적 경험을 공유하는 하나의 집단
> - 코호트 동질성: 같은 시대나 환경을 겪은 사람들은 서로 생각하는 방식, 좋아하는 것, 행동하는 모습이 비슷하다는 뜻

---

## 3.2.1 데이터 정제 (Data Cleaning)

### KEY PAPER 원문

> Missing values are addressed using K-Nearest Neighbors (k = 5) imputation, which preserves the structure of multivariate data and reduces imputation bias compared to mean/mode substitution.

### 본 연구 적용

본 연구는 결측치 처리에 K-최근접 이웃(KNN) 등 대체(imputation) 기법 대신 완전사례분석(Listwise Deletion)을 적용하였다. 이는 본 데이터의 결측이 측정 누락이 아니라 설문 응답 거부 또는 조건부 문항(Skip Logic)에 따른 구조적 미응답이기 때문이다. 무응답 자체가 응답자의 특성(예: 위험성평가 미실시 사업장)을 반영하므로 인위적 대체는 편향을 유발할 수 있다. 동일 원자료를 분석한 박천수(2023, 2024, 2025)의 선행연구도 listwise deletion을 채택하였다.

---

## 3.2.2 특징 공학 (Feature Engineering)

### KEY PAPER 원문

> Categorical Encoding: Categorical variables such as 'Gender' and 'Smoking Status' are encoded using Label Encoding to convert them into numerical format.
>
> Formula for Label Encoding: Encoded Value = Rank of Category / Total Categories

### 본 연구 적용

범주형 인코딩: 인증보유, 공사규모, 발주처, 공사종류 등 범주형 변수는 Label Encoding을 통해 수치형으로 변환하였다.

Label Encoding 공식: Encoded Value = Rank of Category / Total Categories

---

## 3.2.3 특징 스케일링 (Feature Scaling)

### KEY PAPER 원문

> To normalize the features, Standardization is applied to continuous variables (e.g., Age, BMI, Height, Weight), ensuring they have a mean of 0 and a standard deviation of 1. Formula for standardization: z = (x − μ) / σ
>
> Where, x is the feature value, μ is the mean, and σ is the standard deviation of the feature.

### 본 연구 적용

변수 간 단위 차이를 제거하기 위해 연속형 변수(예: 기성공정률, 외국인비율, 리커트 척도 변수)에 표준화(Standardization)를 적용하여 평균 0, 표준편차 1의 분포를 갖도록 변환하였다.

표준화 공식: z = (x − μ) / σ

여기서 x는 변수값, μ는 변수의 평균, σ는 변수의 표준편차이다.

---

## 3.3 통계적 모델링과 머신러닝

### 3.3.1 모델 선정 및 학습

#### KEY PAPER 원문

> We select Random Forest Classifier (RFC) due to its robustness in handling complex datasets with multiple predictors and its ability to capture non-linear relationships.
>
> The RFC algorithm uses decision trees to classify data based on a random subset of features at each node, ultimately combining the trees to improve accuracy and reduce overfitting.
>
> Algorithm for Random Forest Classifier:
> 1. Select m random features from the total p features for each tree.
> 2. Build N decision trees, each based on a bootstrap sample of the training data.
> 3. For each tree, make predictions on the test data, and aggregate the predictions (by majority voting for classification).
>
> Model validation was conducted using 10-fold cross-validation, repeated 3 times. Additionally, an 80/20 train-test split was used to ensure external validation of model generalization.

#### 본 연구 적용

본 연구는 다수의 예측변수를 가진 복잡한 데이터셋을 다루는 데 견고하고 비선형 관계를 포착할 수 있는 랜덤 포레스트 분류기(Random Forest Classifier, RFC)를 주 모델로 선정하였다. RFC 알고리즘은 각 노드에서 무작위로 선택한 변수 부분집합을 기반으로 의사결정 트리를 구축하고, 다수의 트리를 결합하여 분류함으로써 예측 정확도를 높이고 과적합(overfitting)을 방지한다.

비교 모델로는 로지스틱 회귀(Logistic Regression), 서포트 벡터 머신(SVM), 그래디언트 부스팅(GBM)을 함께 학습하여 성능을 비교하였다.

**랜덤 포레스트 분류기 알고리즘:**

1. 전체 p개 변수 중 각 트리마다 m개의 무작위 변수를 선택한다.
2. 학습 데이터의 부트스트랩(bootstrap) 표본을 기반으로 N개의 의사결정 트리를 구축한다.
3. 각 트리가 테스트 데이터에 대해 예측을 수행하고, 다수결 투표(majority voting)로 최종 분류 결과를 산출한다.

여기서 αi는 트리 i의 가중치, fi(x)는 트리 i의 예측 함수이다.

모델 검증은 10겹 교차검증(10-fold cross-validation)을 3회 반복하여 수행하였다. 또한 8:2 학습-테스트 분할(train-test split)을 적용하여 모델의 일반화 성능에 대한 외부 검증을 보장하였다.

---

### 3.3.2 SHAP (SHapley Additive exPlanations)

#### KEY PAPER 원문

> SHAP values provide a method to explain the contribution of each feature to the model's prediction. The SHAP value for a feature is calculated by considering all possible combinations of features and measuring how much the inclusion of each feature changes the prediction.
>
> Where, f(S) is the model prediction using the feature subset S, and N is the set of all features. SHAP values are computed using the TreeExplainer from the SHAP library, which is optimized for decision tree models like Random Forest.

#### 본 연구 적용

SHAP 값은 모델 예측에 대한 각 변수의 기여도를 설명하는 방법을 제공한다. 특정 변수의 SHAP 값은 가능한 모든 변수 조합을 고려하여, 해당 변수의 포함 여부가 예측값을 얼마나 변화시키는지 측정함으로써 산출된다.

여기서 f(S)는 변수 부분집합 S를 사용한 모델의 예측값이며, N은 전체 변수의 집합이다. 본 연구에서는 랜덤 포레스트와 같은 의사결정 트리 기반 모델에 최적화된 SHAP 라이브러리의 TreeExplainer를 사용하여 SHAP 값을 산출하였다.

---

### 3.3.3 평가 지표 (Evaluation Metrics)

#### KEY PAPER 원문

> Accuracy: The proportion of correct predictions to total predictions.
>
> Where, TP, FP, and FN refer to true positives, false positives, and false negatives, respectively. ROC Curve and AUC: The Receiver Operating Characteristic (ROC) curve and Area Under Curve (AUC) are used to evaluate the classifier's ability to distinguish between classes.

#### 본 연구 적용

**정확도(Accuracy)**: 전체 예측 중 올바르게 예측한 비율

**정밀도(Precision), 재현율(Recall), F1 점수(F1-Score)**: 분류 성능을 평가하는 지표로, 특히 클래스 불균형 데이터(본 연구의 사고발생 28.4%)에서 중요하다.

여기서 TP, FP, FN은 각각 참 양성(True Positive), 거짓 양성(False Positive), 거짓 음성(False Negative)을 의미한다.

**ROC 곡선과 AUC**: ROC(Receiver Operating Characteristic) 곡선과 곡선하면적(AUC, Area Under Curve)은 분류기가 두 클래스를 얼마나 잘 구분하는지 평가하는 데 사용된다.

---

## 3.4 교란변수 및 통제변수 (Confounding and Control Variables)

### KEY PAPER 원문

> To address potential confounders such as age, socioeconomic status (proxied via education), and comorbidities (diabetes), these variables were explicitly included as predictors in both traditional regression and machine learning models.

### 본 연구 적용

사고발생에 영향을 미칠 수 있는 사업장의 구조적·물리적 특성을 통제하기 위해, 5개 변수(공사규모, 발주처, 기성공정률, 공사종류, 외국인비율)를 통제변수로 설정하고 전통적 회귀모형과 머신러닝 모형 모두에 예측변수로 명시적으로 포함하였다.

---

## 3.5 통계 검정 방법 (Statistical Testing Methods)

### KEY PAPER 원문

> P-values were calculated using Student's t-tests for continuous variables, chi-square tests for categorical variables, and Wald tests in logistic regression models. All computations were done using R software.

### 본 연구 적용

p값은 연속형 변수에 대해서는 Student's t-검정, 범주형 변수에 대해서는 카이제곱(Chi-square) 검정, 로지스틱 회귀 모형에서는 Wald 검정을 사용하여 산출하였다. 또한 조절효과의 집합적 유의성을 검증하기 위해 우도비 검정(Likelihood Ratio Test)을 추가로 적용하였다. 모든 계산은 Python 환경(statsmodels, scikit-learn, scipy)에서 수행되었다.

---

## 표 1. 본 연구 표본의 변수 기술통계

### KEY PAPER 원문

> Summary of clinical and demographic variables in the study cohort.

### 본 연구 적용

이 표는 분석에 사용된 1,375개 사업장의 13개 연속변수(안전조직수준, 위원회수준, 위험성평가수준, 교육훈련도움, 정리정돈상태, 작업중지권, 작업반장기여, 전문지도, 고용노동부감독, 안전보건공단지원, 기성공정률, 외국인비율 등)에 대해 평균±표준편차, 최소값, 사분위수(Q1, 중앙값, Q3), 최대값을 제시한다.

결과 파일: `results/tables/table1_기술통계.csv`

---

## 표 2. 본 연구 표본의 범주형 변수 및 사고발생 분포

### KEY PAPER 원문

> Distribution of lifestyle factors and chronic illnesses in the cohort. Out of the total 1200 patients, 300 (25%) are non-smokers and 900 (75%) are smokers. Additionally, 150 (12.5%) have diabetes, and 1050 (87.5%) are nondiabetic.

### 본 연구 적용

전체 1,375개 사업장 중 사고가 발생한 사업장은 391개(28.4%), 미발생 사업장은 984개(71.6%)로, 사고 미발생 사례가 더 많은 불균형 구조를 보인다. 인증보유, 공사규모(소·중·대), 발주처, 공사종류(토목·건축·플랜트) 등 범주형 변수의 분포는 표 2와 같다.

결과 파일: `results/tables/table2_범주형분포.csv`

---

## 표 3 — 본 연구에서 생략

### KEY PAPER 원문

> Impact of demographic and clinical variables on health outcomes (Linear Regression Analysis). Age: Each year increase in age is associated with a 0.02 unit increase in the dependent variable (BMI or glucose levels).

### 본 연구 적용

KEY PAPER의 Table 3은 선형회귀(연속형 종속변수 예측)인데, 본 연구의 종속변수 `사고발생`은 이진(0/1)이라 선형회귀가 부적절하므로 **생략**하였다.

---

## 표 4. 사고발생 예측 요인: 로지스틱 회귀 분석

### KEY PAPER 원문

> Predictors of disease presence: logistic regression analysis. Age: For each year increase in age, the odds of the outcome increase by approximately 5%. BMI: An increase in BMI by 1 unit results in a 0.70 increase in the log-odds of the outcome. An interaction term between BMI and smoking status was included. The interaction term (BMI×Smoking) was statistically significant (p = 0.03).

### 본 연구 적용

본 연구는 독립변수를 내부 안전관리(A 그룹)와 현장 안전 행동(B 그룹)으로 분리하여 두 개의 로지스틱 회귀 모형을 추정하였다(표 4-A, 표 4-B).

**표 4-A (A 그룹)**: 통제변수 중 공사규모(OR=1.35, p<0.001), 기성공정률(OR=1.62, p<0.001), 외국인비율(OR=1.28, p<0.001), 공사종류(OR=0.76, p<0.001)가 사고발생에 강하게 영향을 미쳤다. 조절변수 중 고용노동부감독은 OR=1.17(p=0.024)로 양의 효과를 보였는데, 이는 감독이 사고를 유발한다기보다 이미 위험한 사업장에 감독이 우선 배치되는 **선택편향(selection bias)**을 반영한다(Reason의 Swiss Cheese Model 관점).

**표 4-B (B 그룹)**: 정리정돈상태가 OR=0.84(p=0.035)로 유의한 보호효과를 보였다. 즉, 정리정돈이 잘 된 사업장일수록 사고발생 확률이 약 16% 낮았다. 다른 B 그룹 변수(위험성평가수준, 교육훈련도움, 작업중지권, 작업반장기여)는 통계적으로 유의하지 않았다.

또한 본 연구는 외부기관(조절변수)과 독립변수 간의 상호작용항을 모형에 추가하여 조절효과를 검정하였으며, 그 결과는 표 5-A 및 표 5-B에 제시한다.

결과 파일: `results/tables/table4A_A그룹_주효과.csv`, `results/tables/table4B_B그룹_주효과.csv`

---

## 표 5 — KEY PAPER의 Cox 모형 → 본 연구의 조절효과 분석으로 대체

### KEY PAPER 원문

> Survival analysis: effect of age and BMI on event hazard (Cox Proportional Hazards Model). Age: The hazard ratio of 1.05 indicates that for each additional year of age, the hazard of the event occurring increases by 5%.

### 본 연구 적용

KEY PAPER는 시간-사건(time-to-event) 데이터를 활용한 Cox 비례위험 모형으로 연령과 BMI의 위험비(Hazard Ratio)를 산출하였으나, 본 연구의 종속변수(사고발생)는 시점 정보가 없는 이진 변수이므로 Cox 모형을 적용할 수 없다.

본 연구는 KEY PAPER가 Table 4에서 언급한 BMI×흡연 상호작용 검정을 RQ에 맞게 확장하여, **표 5-A(A 그룹 × 조절변수)**와 **표 5-B(B 그룹 × 조절변수)**로 대체하였다.

**표 5-A: A 그룹 × 조절효과**

| 조절변수 | 집합검정 p값 | 유의 상호작용 |
|---|---|---|
| 전문지도 | 0.193 | 없음 |
| **고용노동부감독** | **0.018 ★** | **인증보유×감독 (OR=1.165, p=0.022), 위원회×감독 (OR=0.867, p=0.038)** |
| 안전보건공단지원 | 0.665 | 없음 |

**표 5-B: B 그룹 × 조절효과** — 15개 전부 무의

| 조절변수 | 집합검정 p값 | 유의 상호작용 |
|---|---|---|
| 전문지도 | 0.138 | 없음 |
| 고용노동부감독 | 0.415 | 없음 |
| 안전보건공단지원 | 0.574 | 없음 |

**RQ에 대한 답**: 고용노동부감독만 A 그룹에 유의한 조절효과를 보였으며, B 그룹에는 어떤 외부기관의 조절효과도 관찰되지 않았다 → 가설 지지.

결과 파일: `results/tables/table5A_A그룹_조절효과.csv`, `results/tables/table5B_B그룹_조절효과.csv`

---

## 표 6. 랜덤 포레스트 모델 성능

### KEY PAPER 원문

> Performance of machine learning models in predicting health outcomes: (Random Forest Model) The Random Forest Model has a good predictive performance with an accuracy of 85%. The AUC of 0.89 indicates a strong ability to discriminate between classes.

### 본 연구 적용

랜덤 포레스트 모델은 정확도 73%, AUC 0.72로, 사고 발생 사업장과 미발생 사업장을 구분하는 데 있어 무작위 예측(AUC=0.5)보다 의미 있는 변별력을 보였다. KEY PAPER(정확도 85%, AUC 0.89)보다 수치가 낮은 것은 본 연구가 합성 데이터가 아닌 실제 조사 데이터를 사용하고, 클래스 불균형 처리를 적용하지 않았기 때문이다.

결과 파일: `results/tables/table6_RF성능.csv`

---

## 표 7. 서포트 벡터 머신(SVM) 분류 성능

### KEY PAPER 원문

> Classification performance of support vector machines for health outcome prediction. The SVM Model shows an accuracy of 86%, with an ROC AUC of 0.90. The F1-score of 0.84 suggests a good balance between precision and recall.

### 본 연구 적용

SVM 모델은 정확도 72%, AUC 0.70을 기록하였다. F1 점수는 0.26으로 정밀도와 재현율 간 균형이 낮은데, 이는 사고 미발생(71.6%)이 다수인 불균형 데이터에서 양성 클래스(사고 발생)를 적극적으로 예측하지 못한 결과이다.

결과 파일: `results/tables/table7_SVM성능.csv`

---

## 표 8. 그래디언트 부스팅(GBM) 모델 성능

### KEY PAPER 원문

> Regression performance of gradient boosting machines in predicting continuous health outcomes. The GBM Model has a low RMSE (0.28) and a high R² (0.88). Cross-validation accuracy of 82% confirms the robustness of the model.

### 본 연구 적용

KEY PAPER에서는 GBM을 연속형 결과 예측(회귀)에 사용하여 RMSE와 R²를 보고하였으나, 본 연구의 종속변수는 사고발생 여부(이진)이므로 GBM을 분류 모델로 적용하였다. GBM은 정확도 70%, AUC 0.73으로 4개 모델 중 AUC 기준 최고 성능을 기록하였으며, 교차검증 AUC 평균 0.69로 모델의 안정성을 확인하였다.

결과 파일: `results/tables/table8_GBM성능.csv`

---

## 표 9. 모델 비교

| 모델명 | 정확도 | AUC | 정밀도 | 재현율 | F1 |
|---|---|---|---|---|---|
| 로지스틱 회귀 | 0.71 | 0.71 | 0.45 | 0.17 | 0.24 |
| 랜덤 포레스트 | 0.73 | 0.72 | 0.53 | 0.38 | 0.44 |
| SVM | 0.72 | 0.70 | 0.50 | 0.18 | 0.26 |
| GBM | 0.70 | 0.73 | 0.47 | 0.40 | 0.43 |

머신러닝 모델(RF, GBM)이 로지스틱 회귀보다 소폭 우위를 보이나 격차는 크지 않다. 이는 본 데이터의 선형적 관계가 비교적 강하고, 단순화 트랙에서 클래스 불균형 처리(SMOTENC 등)를 적용하지 않았기 때문으로 해석된다.

결과 파일: `results/tables/table9_모델비교.csv`

---

## 4. 결과 (Results)

### KEY PAPER 원문

> All statistical analyses for this study were conducted using R Software version 4.4.2. The Random Forest model outperformed logistic regression across all metrics, showing higher accuracy (0.85 vs. 0.79) and AUC (0.89 vs. 0.84).
>
> The Fig. 2 provides valuable insights into the health status of the study population, highlighting the range of BMI values. The bell-shaped curve suggests a normal distribution, justifying the use of parametric statistical methods.
>
> This Fig. 3 illustrates the relationships between various health parameters. The matrix reveals how strongly each variable is correlated with one another, with red indicating positive correlations and blue showing negative correlations.
>
> The ROC curve (Fig. 4) analysis is used to evaluate the performance of a predictive model in distinguishing between two classes. The Area Under the Curve (AUC) provides a single value to summarize the model's overall ability to discriminate between the two classes.

### 본 연구 적용

결과를 제시하기에 앞서, 본 연구의 모든 통계 분석은 Python 환경에서 수행되었다. 전통적 통계 모형(로지스틱 회귀)은 statsmodels 라이브러리를, 머신러닝 알고리즘(랜덤 포레스트, 서포트 벡터 머신, 그래디언트 부스팅)은 scikit-learn을, SHAP 분석은 shap 라이브러리를 사용하였다. 분석에 사용된 전체 코드는 재현성 확보를 위해 GitHub 저장소(github.com/geonumul/construction_competition)에 공개하였다.

랜덤 포레스트 모델은 정확도(0.73 vs. 0.71)와 AUC(0.72 vs. 0.71)에서 로지스틱 회귀보다 소폭 우위를 보였다. 다만 본 단순화 트랙에서는 클래스 불균형 처리(SMOTENC 등)를 적용하지 않아 두 모형 간 격차는 크지 않았다.

**그림 2**는 본 연구 표본의 핵심 변수인 기성공정률 분포를 보여주며, 사업장별 공사 진척도가 폭넓게 분포함을 확인할 수 있다. 분포가 한쪽으로 심하게 치우치지 않고 비교적 고르게 퍼져 있어, 후속 분석에서 모수적 통계 방법의 사용을 뒷받침한다.

**그림 3**은 안전조직수준, 위원회수준, 인증보유, 정리정돈상태, 기성공정률, 공사규모 등 주요 변수 간의 상관관계를 보여준다. 빨간색은 양의 상관, 파란색은 음의 상관을 나타낸다. 대부분의 변수 간 상관계수 절댓값이 0.3 미만으로 약한 관계를 보여, **다중공선성 문제가 거의 없음**을 확인할 수 있다.

**그림 4**의 ROC 곡선은 예측 모델이 사고 발생 사업장과 미발생 사업장을 얼마나 잘 구분하는지를 평가한다. 본 연구의 4개 모델은 모두 0.70~0.73 수준으로 무작위 예측(0.5)보다 의미 있게 높은 변별력을 가진다.

---

## 5. 논의 (Discussion)

### KEY PAPER 원문

> This study integrates Random Forest and SHAP to offer both predictive power and interpretability. The model's performance, as evaluated using accuracy (85%), precision (83%), recall (80%), and F1-score (81%), highlights its strong predictive capabilities. By utilizing SHAP, the study offers transparency into the model's decision-making process, revealing that BMI (coefficient = 0.80, p < 0.0001) and Age (coefficient = 0.02, p = 0.038) are the most influential predictors. The SHAP results identify modifiable risk factors (BMI, glucose) as key predictors.

### 본 연구 적용

본 연구는 랜덤 포레스트와 SHAP을 통합하여 건설업 산업재해 데이터에 대한 예측력과 해석 가능성을 동시에 확보하는 프레임워크를 제시하였다. 모델 성능은 정확도 73%, AUC 0.72로, 실제 데이터의 노이즈와 불균형 구조를 고려할 때 실무적으로 유의미한 변별력을 갖는다.

SHAP 분석을 통해 모델의 의사결정 과정을 투명하게 공개한 결과, 기성공정률(평균 |SHAP|=0.078)과 외국인비율(0.053)이 사고 예측에 가장 큰 영향력을 가진 변수로 나타났다. 이는 사고가 개별 안전 행동보다 사업장의 구조적 특성에 더 크게 좌우됨을 시사한다.

한편, 로지스틱 회귀 분석에서는 정리정돈상태(OR=0.84, p=0.035)가 유의한 보호효과를 보였다. 정리정돈은 비용이 크지 않으면서 현장에서 즉시 개선 가능한 행동 변수라는 점에서, KEY PAPER가 BMI를 수정 가능한 위험요인(modifiable risk factor)으로 강조한 것과 동일한 맥락의 발견이다.

조절효과 분석에서는 외부기관 3종 중 고용노동부감독만이, A 그룹(내부 안전관리)에 대해서만 유의한 조절효과를 보였다(LR test p=0.018). B 그룹(현장 안전 행동)에는 어떤 외부기관의 조절효과도 관찰되지 않았다(15쌍 모두 무의). 이는 외부 감독이 제도적·구조적 차원에서만 작동하며, 작업자의 일상 행동에는 직접적 영향을 미치지 못함을 보여준다.

인증보유 × 감독 상호작용의 양의 효과(OR=1.165, p=0.022)는 "감독이 사고를 유발한다"가 아니라, 이미 위험 신호가 누적된 사업장이 감독 대상으로 선정되는 선택편향으로 해석되며, 이는 Reason(1990, 2000)의 Swiss Cheese Model에서 감독을 최후방 방어층으로 위치시킨 것과 이론적으로 부합한다.

SHAP 결과가 식별한 수정 가능한 위험요인(정리정돈상태)과 조절효과 패턴은 고위험 사업장 선별(risk stratification) 및 목표 지향적 안전 관리 개입을 안내하는 데 활용될 수 있다.

---

## 6. 한계점 (Limitations)

### KEY PAPER 원문

> A key limitation of this study is the potential bias introduced by the selection process. Only 1200 patients from an initial 1500 were included after data cleaning. While SHAP values provide valuable insights, the results should be validated across different cohorts. The dataset reflects a single regional center, which may limit generalizability.

### 본 연구 적용

본 연구의 주요 한계점은 다음과 같다. 첫째, 원자료 1,502개 사업장 중 127개가 제외되어 1,375개만 분석에 포함되었으며, 이 선택 과정에서 발생하는 잠재적 편향이 결과의 일반화 가능성에 영향을 미칠 수 있다. 둘째, SHAP 값이 변수 중요도에 대한 유의미한 통찰을 제공하지만, 그 수치가 안전관리 현장에서 실질적으로 타당한지 판단하려면 산업안전 분야의 도메인 지식이 반드시 필요하다. 셋째, 랜덤 포레스트와 SHAP은 검증이 충분하지 않을 경우 과적합에 취약할 수 있다. 넷째, 본 데이터는 2021년 단일 시점의 건설업 횡단면 조사로, 시간에 따른 변화나 다른 업종·국가에 그대로 일반화하기에는 한계가 있다. 마지막으로, 단순화 트랙에서는 클래스 불균형 처리(SMOTENC 등)를 적용하지 않아 머신러닝 모델의 성능이 과소 추정되었을 가능성이 있다.

---

## 7. 향후 연구 (Future Work)

### KEY PAPER 원문

> Future research could explore other advanced machine learning techniques, such as Gradient Boosting or XGBoost, to compare predictive performance. Additionally, investigating complex interactions between features and incorporating more clinical parameters could potentially enhance the model's accuracy. Future research should validate our SHAP-integrated models on independent datasets.

### 본 연구 적용

향후 연구에서는 첫째, 클래스 불균형 처리(SMOTENC 등)를 적용한 정교화 분석을 통해 머신러닝 모델의 예측 성능 향상을 검증할 필요가 있다. 둘째, 본 연구에서 다루지 못한 작업자 개인 수준 변수(경력, 교육 이수 이력, 건강 상태 등)나 현장 환경 변수(계절, 기후 조건 등)를 추가로 포함하여 모델의 정확도와 해석력을 높일 수 있다. 셋째, 본 연구의 SHAP 통합 프레임워크를 제조업·서비스업 등 타 업종이나 다른 연도의 실태조사 데이터에 적용하여 결과의 일반화 가능성과 견고성을 독립적으로 검증해야 한다. 넷째, SHAP Dependence Plot이나 Interaction Value 분석을 추가하여 조절효과의 비선형적 패턴을 심층적으로 탐색할 수 있다.

---

## 8. 결론 (Conclusion)

### KEY PAPER 원문

> The integrated analysis of epidemiological data using both statistical modeling and machine learning techniques has provided valuable insights into the factors influencing patient health outcomes. Through linear regression, logistic regression, and Cox proportional hazards models, BMI and age consistently emerged as significant predictors. These results were reinforced by machine learning models, including Random Forest, SVM, and GBM. The SHAP results identify modifiable risk factors (BMI, glucose) as key predictors. Despite strong performance, external validation is essential.

### 본 연구 적용

본 연구는 통계적 모델링과 머신러닝을 통합하여 건설업 산업재해 발생에 영향을 미치는 요인을 분석하였으며, 전통적 방법론과 현대적 방법론의 상호 보완적 강점을 입증하였다.

로지스틱 회귀 분석에서 정리정돈상태(OR=0.84, p=0.035)는 일관되게 유의한 보호효과를 보이는 수정 가능한 위험요인(modifiable risk factor)으로 확인되었으며, 통제변수인 공사규모·기성공정률·외국인비율은 사고발생에 강한 영향을 미쳤다. 이러한 결과는 랜덤 포레스트, SVM, GBM 등 머신러닝 모델의 SHAP 분석에서도 기성공정률·외국인비율이 상위 변수로 도출됨으로써 교차 검증되었다.

본 연구의 가장 핵심적인 발견은 조절효과 분석에 있다. 외부기관 3종 중 고용노동부감독만이, 내부 안전관리(A 그룹)에 대해서만 유의한 조절효과를 보였으며(LR test p=0.018), 현장 안전 행동(B 그룹)에는 어떤 외부기관도 조절효과를 갖지 않았다. 이는 외부 감독이 제도적 차원에서만 작동한다는 Swiss Cheese Model의 이론적 예측과 부합한다.

본 연구는 의료 역학 분야에서 검증된 SHAP 통합 프레임워크(Qurat Ul Ain & Rather, 2025)를 산업안전 분야에 최초로 이식 적용하여, 통계적 추론(LR)과 예측적 발견(ML+SHAP)의 통합이 건설 현장의 안전 관리 의사결정에도 유효함을 보였다. 향후 클래스 불균형 처리, 타 업종 데이터, 추가 변수 확장을 통한 외부 검증이 필요하다.

---

## 참고 문헌

- Qurat Ul Ain, S., & Rather, K. U. I. (2025). Annals of Epidemiology, 108, 85-91.
- 한국산업안전보건공단 (2021). 제10차 산업안전보건 실태조사 (건설업).
- Reason, J. (1990, 2000). Swiss Cheese Model.
- 박천수 (2023, 2024, 2025). 동일 원자료 선행연구.
- Norman, G. (2010). Likert scales, levels of measurement and the "laws" of statistics. Advances in Health Sciences Education, 15(5), 625-632.
