# AI-Powered Billboard Monitoring System

## Overview

The AI-Powered Billboard Monitoring System is an intelligent computer vision application designed to automate the monitoring and analysis of outdoor advertising billboards. Instead of relying on manual inspections, the system processes captured images to detect billboards, extract advertisement information, estimate billboard dimensions, determine geographic location, and evaluate compliance with predefined advertising regulations.

The system combines modern deep learning, optical character recognition, and large language models to provide a complete automated monitoring solution. It is intended for use by government agencies, municipal authorities, advertising companies, and regulatory organizations that need an efficient method for managing outdoor advertising infrastructure.

## Problem Statement

Monitoring roadside billboards manually is time-consuming, expensive, and prone to human error. Authorities often struggle to verify billboard dimensions, advertisement content, installation location, and regulatory compliance across large geographic regions.

This project addresses these challenges by using Artificial Intelligence to perform automated billboard inspection from captured images.

## Solution

The proposed system automatically performs the following tasks:

* Detects billboards using a trained YOLO object detection model.
* Crops the detected billboard region from the original image.
* Extracts advertisement text using OCR.
* Estimates billboard dimensions using computer vision techniques.
* Retrieves GPS coordinates from image metadata when available.
* Determines the physical location through reverse geocoding.
* Uses Google Gemini AI to analyze billboard content and evaluate regulatory compliance.
* Generates structured compliance reports for authorities.

## Features

* Automatic billboard detection
* Image preprocessing
* OCR-based text extraction
* Billboard size estimation
* GPS metadata extraction
* Reverse geolocation
* AI-powered compliance analysis
* Structured JSON report generation
* REST API using Flask
* Modular architecture for future expansion

## Technology Stack

### Programming Language

* Python

### Backend

* Flask
* Flask-CORS

### Artificial Intelligence

* YOLO
* Roboflow Inference SDK
* Google Gemini AI

### Computer Vision

* OpenCV

### OCR

* EasyOCR
* Tesseract OCR

### Location Services

* EXIF Metadata
* Nominatim Reverse Geocoding

### Data Processing

* NumPy
* Pillow

## System Workflow

1. User uploads a billboard image.
2. YOLO detects billboard objects.
3. The detected billboard region is cropped.
4. OCR extracts advertisement text.
5. Computer vision estimates billboard dimensions.
6. GPS metadata is extracted if available.
7. Reverse geocoding identifies the location.
8. Gemini AI evaluates advertisement compliance.
9. A structured compliance report is generated and returned.

## Applications

* Smart city infrastructure monitoring
* Municipal billboard inspections
* Outdoor advertising management
* Regulatory compliance verification
* Urban planning
* Digital governance
* Advertising analytics

## Future Enhancements

* Live CCTV integration
* Drone-based billboard monitoring
* Real-time violation alerts
* Automatic violation ticket generation
* Dashboard with GIS map visualization
* Mobile application support
* Multi-language OCR
* Cloud deployment with scalable processing

## Conclusion

The AI-Powered Billboard Monitoring System demonstrates how artificial intelligence, computer vision, and natural language processing can be integrated to automate billboard inspection. By reducing manual effort and improving inspection accuracy, the system provides an efficient solution for monitoring outdoor advertisements while supporting regulatory compliance and smart city initiatives.
