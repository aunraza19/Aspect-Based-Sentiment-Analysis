# 📈 PyABSA Aspect-based Sentiment Analysis Application
Link to Demo: (https://huggingface.co/spaces/aun09/Aspect-Based-Sentiment-Analysis)

## Table of Contents
1.  [Project Overview](#1-project-overview)
2.  [Problem Covered](#2-problem-covered)
3.  [Solution](#3-solution)
4.  [Key Functionalities](#4-key-functionalities)
5.  [Technical Architecture](#5-technical-architecture)
    * [File Structure](#file-structure)
    * [Core Components](#core-components)
    * [Dependencies](#dependencies)
6.  [Local Setup and Running](#6-local-setup-and-running)
7.  [How to Use the Application](#8-how-to-use-the-application)
8.  [Customization and Extension](#9-customization-and-extension)

---

## 1. Project Overview

This project presents a streamlined web application built with Gradio that leverages the powerful PyABSA library. Its primary purpose is to demonstrate **Multilingual Aspect-based Sentiment Analysis (ATEPC)**. Unlike traditional sentiment analysis which provides an overall sentiment for a text, ATEPC dives deeper, identifying specific aspects (entities or attributes) within a sentence and determining the sentiment expressed towards each of them individually.

This refined version of the application focuses exclusively on ATEPC and provides a curated list of datasets for demonstration.

## 2. Problem Covered

In many real-world scenarios, understanding the overall sentiment of a document or review is insufficient. For instance, a customer might write: "The restaurant's **food** was amazing, but the **service** was incredibly slow." A general sentiment analyzer might classify this review as "neutral" or "mixed."

The problem this project addresses is the need for **fine-grained sentiment understanding**. Businesses and researchers often need to know:
* What specific entities or attributes are being discussed?
* What is the sentiment (positive, negative, neutral) towards *each* of these aspects?

This level of detail is crucial for:
* **Customer Feedback Analysis:** Pinpointing exactly what customers like or dislike about products/services (e.g., "battery life" of a laptop, "ambiance" of a restaurant).
* **Market Research:** Understanding public opinion on different features of a new product.
* **Competitive Analysis:** Comparing specific strengths and weaknesses across competitors based on reviews.

## 3. Solution

This application provides an intuitive solution to the problem of aspect-based sentiment analysis using the following components:

* **PyABSA Library:** At its core, the application uses PyABSA (Python Aspect-based Sentiment Analysis), a state-of-the-art library for various ABMS tasks. It comes with pre-trained models, making it versatile for diverse textual data.
* **Gradio Interface:** A user-friendly web interface is built using Gradio, allowing users to:
    * Input custom sentences for analysis.
    * Select from a predefined set of popular datasets to automatically load random examples.
    * View the extracted aspects, their sentiments, confidence scores, and positions in a clear tabular format.
* **Modular Code Structure:** The code is organized into logical Python files (`app.py`, `models.py`, `utils.py`) to enhance readability, maintainability, and reusability, making it easier to extend or integrate into larger projects.

## 4. Key Functionalities

* **Aspect Term Extraction (ATE):** Automatically identifies and extracts relevant aspect terms (e.g., "food", "service", "battery life") from a given input sentence.
* **Aspect Sentiment Classification (APC):** Determines the sentiment polarity (Positive, Negative, Neutral) expressed towards each extracted aspect term.
* **Multilingual Support:** Utilizes a `multilingual` PyABSA checkpoint, enabling analysis across various languages without requiring language-specific models.
* **Example Loading:** Users can choose from a select list of well-known datasets (`Laptop14`, `Restaurant14`, `SemEval`, `Twitter`, `TShirt`) to load random example sentences, facilitating quick demonstrations and exploration.
* **Interactive Web UI:** Provides a simple and responsive web interface via Gradio for easy interaction.

## 5. Technical Architecture

The project follows a modular Python structure, specifically adapted for seamless deployment on Hugging Face Spaces.

### File Structure
```bash
Aspect-Based-Sentiment-Analysis
├── src/
│   ├── utils.py         # Helper functions for loading dataset examples
│   ├── models.py        # Handles PyABSA model initialization and dataset
│   └── app.py           # Main Gradio application script
├── requirements.txt     # Lists all required Python packages
└── README.md            # This comprehensive README file
```
**Note:** For simplicity and ease of deployment on Hugging Face Spaces, all core Python files are placed directly in the repository root. This avoids complex import paths and ensures the Gradio application is automatically detected.

### Core Components

* **`app.py`**:
    * This is the entry point of the Gradio application.
    * It defines the entire web user interface using `gradio.Blocks()`.
    * It integrates the `run_atepc_inference` function, which orchestrates the sentiment analysis by calling the PyABSA model.
    * Manages the display of input text and the resulting DataFrame of aspects and sentiments.
* **`models.py`**:
    * Responsible for initializing the `AspectTermExtraction` model from PyABSA.
    * Handles the `download_all_available_datasets()` call, ensuring that all necessary PyABSA datasets and models are downloaded upon the first run.
    * Includes robust error handling and logging during model initialization to provide clear feedback.
* **`utils.py`**:
    * Contains helper functions, specifically `load_atepc_examples()`.
    * This function reads and cleans sentences from the PyABSA datasets, preparing them for use as interactive examples in the Gradio interface. It handles dataset file detection and ensures data is properly formatted.
* **`requirements.txt`**:
    * A standard Python file that lists all external libraries required for the project to run. This is essential for both local setup and cloud deployment environments like Hugging Face Spaces.

### Dependencies

The project relies on the following key Python libraries, specified in `requirements.txt`:

* `pyabsa`: The core library for Aspect-based Sentiment Analysis.
* `gradio`: Used to create the interactive web user interface.
* `pandas`: Utilized for efficient data handling and displaying tabular results (DataFrames).

## 6. Local Setup and Running

To run this application on your local machine (e.g., in PyCharm):

1.  **Clone the Repository or Create Files:**
    If you have the project files locally from a previous step, ensure they are organized as described in the [File Structure](#file-structure) section. If you are starting fresh, create the project directory and populate the files (`app.py`, `models.py`, `utils.py`, `requirements.txt`, `README.md`) with the provided code.

2.  **Create a Python Virtual Environment (Highly Recommended):**
    A virtual environment isolates your project dependencies, preventing conflicts with other Python projects.
    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment:**
    * **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install Required Packages:**
    With your virtual environment activated, install all dependencies:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the Application:**
    Navigate to the root of your project directory (where `app.py` is located) and run:
    ```bash
    python app.py
    ```
    The first time you run it, `PyABSA` will download necessary models and datasets. This may take some time. Once completed, Gradio will provide a local URL (e.g., `http://127.0.0.1:7860/`) in your terminal. Open this URL in your web browser to access the application.


## 7. How to Use the Application

The Gradio interface provides a simple way to perform Aspect-based Sentiment Analysis:

1.  **Input a Sentence:**
    * Type or paste a sentence into the "Input Sentence" text box. For example: "The pizza was delicious, but the delivery was very slow."

2.  **Select Dataset (Optional for Examples):**
    * Below the input box, you'll see a radio button group titled "Select Dataset (for random examples):".
    * If you leave the "Input Sentence" box empty, selecting a dataset (e.g., "Restaurant14", "Laptop14") will automatically load a random example sentence from that specific dataset into the input box. This is useful for quickly seeing the model in action.

3.  **Analyze Aspects!:**
    * Click the "Analyze Aspects!" button.

4.  **View Results:**
    * **Analyzed Sentence:** The text area below the button will display the exact sentence that was processed (either your input or the loaded example).
    * **Aspect Prediction Results:** A table will appear, showing:
        * **Aspect:** The specific term identified (e.g., "pizza", "delivery").
        * **Sentiment:** The sentiment polarity towards that aspect (e.g., "Positive", "Negative").
        * **Confidence:** The model's confidence score for that prediction (a value between 0 and 1).
        * **Position:** The start and end character indices of the aspect term in the original sentence.

## 8. Customization and Extension

This project provides a solid foundation that you can extend:

* **Add More Datasets:** If you wish to include more ATEPC datasets from PyABSA's available list (`ATEPC.ATEPCDatasetList()`), you can modify the `DESIRED_ATEPC_DATASETS` list in `app.py`. Remember to verify the dataset names are exact matches.
* **Integrate Other PyABSA Tasks:** While this version focuses on ATEPC, you could reintroduce Aspect Sentiment Triplet Extraction (ASTE) or other PyABSA tasks by adding back the relevant code from the earlier versions of the project.
* **UI Customization:** Gradio offers extensive options for styling and layout. You can modify the `gr.Blocks()` section in `app.py` to change colors, fonts, component arrangements, and more to suit your aesthetic preferences.
* **Model Fine-tuning:** For domain-specific sentiment analysis, you might consider fine-tuning PyABSA models on your own custom datasets. Refer to the official PyABSA documentation for guides on training.
## 📄 License
MIT License. Feel free to fork and build upon this for research or academic use.

## 👤 Author
Made by Aun Raza

If you use this project, feel free to ⭐ star it.
---
