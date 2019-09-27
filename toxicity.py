import os
import subprocess
from service.flask_service import main as FlaskApp

def main():
    subprocess.call("sudo npm start &", shell=True) # Start Web Application (NodeJS)"
    flask_server = FlaskApp() # Start Flask Server (Python)

if __name__ == "__main__":
    main()
