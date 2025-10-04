# 🌍 EXO AI - The Exoplanet Analyzer

**AI-Powered Exoplanet Detection & Habitability Analysis**

Built for NASA Space Apps Challenge 2025 🚀

---

## What's This About?

Ever wondered if there are Earth-like planets out there? So did we! 

EXO AI is our answer to NASA's challenge of finding exoplanets using artificial intelligence. We built a dual-AI system that not only detects exoplanets from telescope data but also figures out if they could potentially support life. Pretty cool, right?

---

## What It Does

- **Detects Exoplanets**: Our ML model analyzes data from NASA's Kepler, K2, and TESS missions to classify planets as CONFIRMED, CANDIDATE, or FALSE POSITIVE with 78.5% accuracy
- **Checks Habitability**: For confirmed planets, we calculate their potential to support life based on temperature, size, and orbit
- **Two Ways to Use It**: Enter data manually for a single planet, or upload a CSV file to analyze hundreds at once

---

## The Tech Stack

- **Machine Learning**: RandomForest models trained on 20,000+ exoplanet candidates
- **Frontend**: Streamlit (because who has time for HTML/CSS during a hackathon?)
- **Data**: Real NASA mission data from Kepler, K2, and TESS archives
- **Science**: Earth Similarity Index (ESI) and Habitable Zone Distance (HZD) calculations

---

## Try It Out!

🔗 **Live Demo**: [https://exoai-explorers.streamlit.app](https://exoai-explorers.streamlit.app)

### Running Locally

