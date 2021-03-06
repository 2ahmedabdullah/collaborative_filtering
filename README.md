## What are Recommendation Engines?

A recommendation engine filters the data using different algorithms and recommends the most relevant items to users. It first captures the past behavior of a customer and based on that, it recommends products which the users might be likely to buy.

![1](fact.png)

## Goal

Guess the missing values in the Rating Matrix

## About the Dataset

Movie Lens Kaggle Data 100k

## The Approach

User-User Collaborative Filtering using Matrix Factorization.

This algorithm first finds the similarity score between users. Based on this similarity score, it then picks out the most similar users and recommends products which these similar users have liked or bought previously.
