from flask import Flask, send_file
import main 
import io

app = Flask(__name__)

@app.route("/")
def get_image():
    try:
        # Return the image as a Flask response
        return send_file(io.BytesIO(main.img_byte_array), mimetype="image/png")

    except Exception as e:
        error_message = f"Error: {e}"
        return error_message, 500  
    

