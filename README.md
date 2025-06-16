Installation and Usage Instructions
=================================================

INSTALLATION
-----------

1. Clone the repository:
   git clone https://github.com/Syfrsam/Web-Image-Filter-OpenCV.git

2. Create a virtual environment:
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate

3. Install required packages:
   pip install -r requirements.txt

4. Run the application:
   python simple_image_filter.py

5. Access the web interface:
   Open your browser and go to http://127.0.0.1:5000/


USAGE
-----

1. Upload an image via the web interface
2. Select a filter to apply to your image
3. Adjust parameters (brightness, contrast, saturation) using the sliders
4. View the processed image in real-time
5. Download the processed image if desired


AVAILABLE FILTERS
----------------

- Edge Detection
- Sharpen
- Blur
- Grayscale
- Sepia
- Invert
- Emboss


ADJUSTMENT PARAMETERS
-------------------

- Brightness: Adjusts the overall lightness/darkness (-100 to 100)
- Contrast: Adjusts the difference between light and dark areas (-100 to 100)
- Saturation: Adjusts the intensity of colors (-100 to 100)


TROUBLESHOOTING
--------------

- If the application fails to start, check if all required packages are installed
- Make sure ports (default 5000) are not being used by other applications
- Verify that Python version 3.6 or higher is installed
- Check if Flask is properly installed in your environment


FILE STRUCTURE
-------------

- simple_image_filter.py: Main application file
- templates/: Directory containing HTML templates
- requirements.txt: List of required Python packages
- .gitignore: Git configuration file


REQUIREMENTS
-----------

- Python 3.6+
- Flask
- Pillow (Python Imaging Library)
- Other dependencies listed in requirements.txt


COLLABORATION WITH: Hanaf1
-------
For issues or questions, please visit:
https://github.com/Hanaf1/FilterImageWeb
