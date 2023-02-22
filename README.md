# Sentiment Analysis of Mobile Money Service M-PESA Using Twitter Data.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/ericgwash/M-PESA-_SENTIMENT-ANALYSIS/main/app.py)
## Deploying the model
M-Pesa is a mobile money service that was first launched in Kenya in 2007. It allows users to store and transfer money, pay bills, and purchase goods and services using their mobile phones. M-Pesa is a popular alternative to traditional banking in many African countries, where a large proportion of the population is unbanked. The service has expanded to other countries in Africa, Asia, and Europe, and has become a model for other mobile money services around the world. M-Pesa has been credited with revolutionizing mobile banking and transforming the way people in developing countries access financial services.Mobile money service M-pesa is the most popular in Kenya, providing financial services to millions of people who previously lacked access to traditional banking services. However, despite the widespread adoption of M-pesa, the sentiment of the M-pesa users is not well understood especially online on twitter. This project aims to fill this gap by analyzing and accurately predicting a tweet's sentiment about M-pesa in Kenya as it is important to understand the sentiment of tweets to gauge public opinion and potentially make improvements.Sentiment analysis is a process of determining the sentiment or emotion expressed in a piece of text.
The sentiment expressed in tweets related to mobile money service M-pesa is relevant in understanding the public opinion towards the service. This information can be useful for the service provider in improving their service, addressing any negative sentiment expressed by users and to research real customer needs and assess brand reputation. Sentiment analysis is leveraged to identify the polarity of information (positive vs. negative), emotion (anger, happiness, sadness, etc.), and intention (e.g., interested and not interested).The tweets data was scraped from twitter between the dates of "2018-01-01" and "2023-02-09".

### Installation

Clone the repository and navigate to the directory.

```bash
git clone https://github.com/ericgwash/M-PESA-_SENTIMENT-ANALYSIS.git
cd M-PESA-_SENTIMENT-ANALYSIS
```
Install the required packages using pip.

```
pip install -r requirements.txt

```

### Usage
Run the Streamlit app.

```
streamlit run app.py
```
The app should now be running on http://localhost:8501.

Files
* app.py: Main Streamlit app file.
* tokenizer.json: Tokenizer file.
* model.h5: Trained model file([https://drive.google.com/drive/folders/18j3-yUjKG30RPbR-2L1NibyInIiyHCGt?usp=sharing](https://drive.google.com/file/d/18WVKvPe96p5EJWl0CLFlLHNA_GxaOPw2/view?usp=sharing)).
* requirements.txt: Required Python packages.

### License
This project is licensed under the MIT License - see the LICENSE file for details.
