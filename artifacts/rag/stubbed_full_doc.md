# Mayo Clinic Diabetes Diagnosis and Treatment Summary

This document is a cleaned experimental stand-in for a full diabetes guidance document.

## Diagnosis

Common diagnostic tests for diabetes include:

- **Hemoglobin A1c (HbA1c)**:
  - Below 5.7% is generally considered in the non-diabetic range.
  - 5.7% to 6.4% is commonly treated as prediabetes.
  - 6.5% or higher is consistent with diabetes.
- **Blood glucose testing**:
  - Fasting plasma glucose of 126 mg/dL or higher is consistent with diabetes, but only when the measurement is explicitly fasting.
  - Current/random plasma glucose of 200 mg/dL or higher should be interpreted using the random glucose threshold in this project.

## Clinical context

Risk can be higher in the presence of:

- older age
- elevated body mass index (BMI)
- hypertension
- cardiovascular disease history
- smoking-related risk factors

## Interpretation guidance

When making a binary diabetes classification:

- prioritize strong diagnostic markers such as HbA1c and blood glucose
- interpret `blood_glucose_level` as a current/random glucose value unless explicitly stated otherwise
- use age, BMI, hypertension, heart disease, and smoking history as supporting context
- prefer a conservative NO only when the major diabetes markers are clearly below diagnostic levels

## Experimental use

This file is used as a controlled full-document RAG condition. It bypasses retrieval search and supplies a stable full context directly to the LLM so retrieval quality can be separated from reasoning quality.
