# SAR Image Colorization Web Application  

This project is a web application designed to colorize Synthetic Aperture Radar (SAR) images, transforming monochrome data into realistic colored representations. The application combines deep learning models (U-Net and GAN) with a React.js frontend to enable real-time visualization and enhanced interpretability of SAR data.  

## Features  
- **Real-Time Visualization:** Upload SAR images and get instant colorized outputs.  
- **Deep Learning Integration:** Uses U-Net and GAN architectures for high-quality image colorization.  
- **User-Friendly Interface:** Responsive and intuitive design built with React.js.  

## Tech Stack  
### Frontend:  
- React.js  
- HTML5, CSS3   

### Backend & Model Integration:  
- Python  
- TensorFlow/Keras  
- Flask 

## Project Structure  
```bash
  ├── src/ # React frontend source code
│ ├── components/ # Reusable React components
│ ├── services/ # API integration for model inference
│ └── styles/ # CSS or styling frameworks
├── backend/ # Deep learning model and API code
│ ├── model/ # Pre-trained U-Net and GAN models
│ ├── api/ # Flask or FastAPI code for serving the model
│ └── utils/ # Utility scripts for preprocessing
├── public/ # Static assets for React app
├── Dockerfile # Docker configuration file (if used)
├── package.json # React dependencies
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```

## Installation  
### Prerequisites:  
- Node.js and npm  
- Python 3.x and pip  

### Steps:  
1. Clone the repository:  
  ```bash
    git clone https://github.com/your-repo/sar-colorization-app.git
    cd sar-colorization-app  
  ```
2. Install Frontend Dependencies:
  ```bash
    cd src  
    npm install
  ```
3. Install Backend Dependencies:
  ```bash
    cd backend  
    pip install -r requirements.txt
  ```
4. Run the Backend:
  ```bash
    python app.py
  ```
5. Run the Frontend:
  ```bash
    cd frontend  
    npm start
  ```  
### Team Members
- Manya – Web Application Development Lead
- Palak Khanna – Deep Learning Model Specialist
- Shanvi – Data Preprocessing and Evaluation
