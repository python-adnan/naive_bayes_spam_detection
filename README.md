# Naive Bayes Spam Detection

This repository contains a Python implementation of the Naive Bayes algorithm built from scratch to classify emails as either spam or ham (non-spam). This project was developed as part of the Probability and Statistics for Machine Learning and Data Science course[cite: 1].

## Project Overview

The Naive Bayes algorithm relies on Bayes' Theorem to determine the probability that a data point belongs to a specific class[cite: 1]. In this binary classification problem, spam emails are labeled as 1, and ham emails are labeled as 0[cite: 1].

The mathematical foundation of this model aims to calculate the following probability[cite: 1]:
$$P(\text{spam} \mid \text{email}) = \frac{P(\text{email} \mid \text{spam}) \cdot P(\text{spam})}{P(\text{email})}$$

By making the "naive" assumption that the probability of each word appearing in an email is independent of the others, we can apply the product rule to calculate the likelihood of an email being spam or ham[cite: 1].

## Key Features

*   **Text Preprocessing:** Implements tokenization and the removal of stopwords and punctuation using the Natural Language Toolkit (NLTK)[cite: 1].
*   **From-Scratch Implementation:** Calculates word frequencies, class frequencies, and conditional probabilities without relying on high-level machine learning libraries like Scikit-Learn[cite: 1].
*   **Numerical Stability Handling:** Includes an enhanced version of the algorithm that uses log probabilities to solve the underflow problem caused by multiplying thousands of small floating-point numbers[cite: 1].
*   **Performance Metrics:** Evaluates the model using Accuracy, Precision, and Recall metrics[cite: 1]. 

## Results

The standard Naive Bayes implementation achieved a baseline accuracy, but suffered from false positives due to numerical underflow on long emails[cite: 1]. By implementing log probabilities, the model transforms the large product calculation into a summation[cite: 1]:
$$\log \left(P(\text{class}) \cdot P(\text{email} \mid \text{class}) \right) = \log \left(P(\text{class}) \right) + \log \left( P(\text{email} \mid \text{class}) \right)$$

This mathematical enhancement improved the model's accuracy to 99.21% on the test set and raised the precision to 98.42%, drastically reducing the number of ham emails mistakenly sent to the spam folder[cite: 1].

## Dependencies

To run this notebook, you will need the following libraries[cite: 1]:
*   numpy
*   pandas
*   nltk

## Dataset

The project utilizes an unbalanced dataset consisting of 5,728 emails, where approximately 23.88% are spam[cite: 1].