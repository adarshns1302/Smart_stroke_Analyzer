# 🏏 Smart Stroke Analyzer
**A computer-vision-powered cricket batting stroke analysis system using YOLOv8 and MediaPipe, with AI-driven pose comparison and detailed player feedback.**

# 🚀 Project Overview
**Smart Stroke Analyzer** is an intelligent computer vision application designed to analyze cricket batting strokes from raw video footage. It uses:

- **YOLOv8** for ball and batsman detection
- **MediaPipe** for advanced pose estimation
- **Custom ResNet18 stroke classifier** for stroke recognition
- **AI-based pose comparison** to generate actionable suggestions
- **Interactive Streamlit dashboard** to present results and download PDF reports

This end-to-end pipeline helps players, coaches, and enthusiasts improve batting technique by analyzing key moments 
like the bat-ball contact point, visualizing pose, and providing performance insights.

# ✨ Features:-
✅ Automatic detection of ball, batsman, and contact frame  
✅ Trajectory tracking of the ball  
✅ Precise pose estimation on contact frame  
✅ Pre-trained ResNet18 classifier for stroke type prediction  
✅ AI-based pose feedback comparing player pose with reference poses   
✅ Easy-to-use Streamlit interface with re-encoded playback  

# 🛠️ Installation
**Clone the repository:**

```bash
git clone https://github.com/adarshns1302/Smart_stroke_Analyzer.git
cd Smart_stroke_Analyzer
```

Set up a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate   on Windows
```

Install dependencies:
```bash
pip install -r requirements.txt
```

# 🏃 Usage
1. Place your input videos in data/raw_videos/.
2. Run the Streamlit app:
```bash
cd feedback_app
streamlit run app.py
```

# 📊 Example Results
Here’s what you get:
1. Annotated video with ball & batsman detection
2. Trajectory tracking overlay
3. Contact frame with pose landmarks
4. Stroke type label with confidence score
5. Suggestions on improving batting technique

![image](https://github.com/user-attachments/assets/744f027e-5cca-4f21-bed0-5ddd01ba76d9)
![image](https://github.com/user-attachments/assets/01477997-7313-4a40-9ec8-1e0499241918)
![image](https://github.com/user-attachments/assets/c02d1089-c4e1-44d7-9c79-991cdea9c589)
![image](https://github.com/user-attachments/assets/16764e23-1789-4b65-959a-98e00e22906e)
![image](https://github.com/user-attachments/assets/29f7be2f-cd82-478b-b3b5-d2311bd4a63c)

# 🤝 Contributing
**Pull requests are welcome! Feel free to open issues for:**
1. Bugs
2. Feature suggestions
3. Code improvements

# 📄 License:
This project is licensed under the [MIT](https://choosealicense.com/licenses/mit/) License.

🌟 Acknowledgments
1. Ultralytics YOLOv8
2. Google MediaPipe
3. PyTorch
4.All open-source contributors who made this work possible!
