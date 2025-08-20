import os
import time
import base64
import tempfile
import streamlit as st
import streamlit.components.v1 as components
import paramiko
import subprocess
import pandas as pd
from gtts import gTTS
import pyautogui
import pywhatkit
from twilio.rest import Client
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import tweepy
from instagrapi import Client as InstaClient
import requests
import google.generativeai as genai
import random
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
from cvzone.HandTrackingModule import HandDetector
import boto3
from PIL import Image, ImageDraw, ImageFont
import psutil
from googlesearch import search
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import seaborn as sns


# ------------------ UTILITY FUNCTIONS ------------------
def speak(text):
    tts = gTTS(text)
    tts.save("speak.mp3")
    with open("speak.mp3", "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        audio_html = f"""
            <audio autoplay>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
    os.remove("speak.mp3")

def extract_number(text):
    words_to_digits = {
        "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7,
        "eight": 8, "nine": 9, "ten": 10
    }
    if text.isdigit():
        return int(text)
    return words_to_digits.get(text.lower(), 1)

def ping_host(ip):
    try:
        result = subprocess.run(["ping", "-c", "1", ip], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
        return result.returncode == 0
    except Exception:
        return False

def ssh_connect(ip, username, password):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(ip, username=username, password=password, timeout=10)
        st.session_state["ssh"] = ssh
        st.success("SSH connected successfully. 🎉")
        return True
    except paramiko.AuthenticationException:
        st.error("Authentication failed. Please check your username and password.")
    except paramiko.SSHException as e:
        st.error(f"SSH error: {e}")
    except Exception as e:
        st.error(f"Failed to connect: {e}")
    return False

def run_command(ssh, command):
    try:
        stdin, stdout, stderr = ssh.exec_command(command, timeout=30)
        return stdout.read().decode(), stderr.read().decode()
    except Exception as e:
        return "", f"Error executing command: {e}"

@st.cache_resource
def setup_gemini():
    api_key = "AIzaSyD5G58h5snx2FPUntIvbNphuFV1WUsTySI" # Consider using Streamlit Secrets for API keys
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("models/gemini-1.5-flash-latest")

def get_mock_prices(source, destination):
    return {
        "Rapido": round(random.uniform(40, 100), 2),
        "Ola": round(random.uniform(60, 120), 2),
        "Uber": round(random.uniform(50, 110), 2)
    }

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

@st.cache_data
def get_ram_info():
    virtual_mem = psutil.virtual_memory()
    return {
        "total": virtual_mem.total / (1024 ** 3),
        "available": virtual_mem.available / (1024 ** 3),
        "usage": 100 * (1 - virtual_mem.available / virtual_mem.total)
    }

@st.cache_data
def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

@st.cache_data(show_spinner=False)
def scrape_website_data(url, max_pages=10, domain_only=True):
    visited = set()
    to_visit = [url]
    data = []

    while to_visit and len(visited) < max_pages:
        current_url = to_visit.pop(0)
        if current_url in visited:
            continue

        try:
            response = requests.get(current_url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')

            for script in soup(["script", "style"]):
                script.decompose()

            page_data = {
                "url": current_url,
                "title": soup.title.string if soup.title else "No title",
                "text": soup.get_text(separator=' ', strip=True),
                "links": []
            }

            data.append(page_data)
            visited.add(current_url)

            for link in soup.find_all('a', href=True):
                absolute_url = urljoin(current_url, link['href'])
                if domain_only:
                    if urlparse(absolute_url).netloc == urlparse(url).netloc:
                        page_data["links"].append(absolute_url)
                        if absolute_url not in visited and absolute_url not in to_visit:
                            to_visit.append(absolute_url)
                else:
                    page_data["links"].append(absolute_url)
                    if absolute_url not in visited and absolute_url not in to_visit:
                        to_visit.append(absolute_url)

            time.sleep(0.5) # Be polite with delay between requests
        except Exception as e:
            st.warning(f"Could not scrape {current_url}: {e}")
            continue
    return data

@st.cache_data
def save_data(data, format='csv'):
    df = pd.DataFrame(data)
    if format == 'csv':
        return df.to_csv(index=False).encode('utf-8')
    elif format == 'json':
        return df.to_json(orient='records').encode('utf-8')
    elif format == 'excel':
        # Create a BytesIO object to save to in-memory
        from io import BytesIO
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        output.seek(0)
        return output.getvalue()


# Modified cached function: now takes a file path or BytesIO object, not a widget
@st.cache_data
def load_titanic_data_from_path(file_path_or_buffer):
    return pd.read_csv(file_path_or_buffer)

@st.cache_data
def load_alexa_data_from_path(file_path_or_buffer):
    return pd.read_csv(file_path_or_buffer, sep='\t')

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Ultimate Multi-Tool Dashboard", layout="wide")

# Replace your existing CSS with this updated version
st.markdown("""
<style>
/* Main color palette - dark theme */
:root {
    --primary: #4e73df;
    --secondary: #6c757d;
    --success: #28a745;
    --info: #17a2b8;
    --warning: #ffc107;
    --danger: #dc3545;
    --light: #f8f9fa;
    --dark: #343a40;
    --text-dark: #212529;  /* For input/options */
    --text-light: #ffffff;  /* For regular text */
    --bg-dark: #1a1a2e;    /* Dark background */
    --bg-light: #2a2a3a;   /* Lighter elements */
}

/* Base styling - dark theme */
body {
    background-color: var(--bg-dark);
    color: var(--text-light) !important;
    font-family: 'Nunito', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    font-size: 1.05rem;  /* Slightly larger base font size */
}

/* Main container styling */
.main .block-container {
    background-color: var(--bg-dark);
    border-radius: 0.35rem;
    box-shadow: 0 0.15rem 1.75rem 0 rgba(0, 0, 0, 0.3);
    padding: 2rem;
    margin-top: 1rem;
    margin-bottom: 1rem;
    color: var(--text-light) !important;
}

/* Make all regular text white and larger */
p, h1, h2, h3, h4, h5, h6, label, .stMarkdown, 
.streamlit-expanderHeader, .streamlit-expanderContent,
.stAlert, .stDataFrame, .stTable {
    color: var(--text-light) !important;
    font-size: 1.05rem;  /* Increased base size */
}

h1 { font-size: 2.2rem !important; }
h2 { font-size: 1.8rem !important; }
h3 { font-size: 1.5rem !important; }
h4 { font-size: 1.3rem !important; }

/* Input fields and options - keep dark text */
.stTextInput>div>div>input, 
.stTextArea>div>textarea,
.stSelectbox>div>div>div,
.stNumberInput>div>div>input,
.stDateInput>div>div>input,
.stSelectbox>div>div>div>div,
.stRadio>div>label, 
.stCheckbox>label,
.stMultiSelect>div>div>div,
.stSlider>div>div>div,
.stFileUploader>label {
    color: var(--text-dark) !important;
    background-color: white !important;
    font-size: 1rem;  /* Slightly smaller for inputs */
}

/* Select dropdown options */
.stSelectbox>div>div>div>div>div {
    color: var(--text-dark) !important;
    background-color: white !important;
}

/* Sidebar styling */
.st-emotion-cache-vk33gh {
    background: linear-gradient(180deg, var(--primary) 10%, #224abe 100%);
    color: var(--text-light) !important;
    padding: 0 1.5rem 2rem;
}

/* Button styling */
.stButton>button {
    border-radius: 0.35rem;
    border: none;
    color: white !important;
    background-color: var(--primary);
    padding: 0.5rem 1rem;
    font-weight: 600;
    transition: all 0.2s;
    font-size: 1rem;
}

.stButton>button:hover {
    background-color: #2e59d9;
    transform: translateY(-1px);
    box-shadow: 0 0.125rem 0.25rem 0 rgba(0, 0, 0, 0.3);
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
}

.stTabs [data-baseweb="tab"] {
    padding: 0.5rem 1rem;
    border-radius: 0.35rem;
    background-color: var(--bg-light);
    transition: all 0.2s;
    color: var(--text-light) !important;
    font-size: 1rem;
}

.stTabs [data-baseweb="tab"]:hover {
    background-color: #3a3a4a;
}

.stTabs [aria-selected="true"] {
    background-color: var(--primary);
    color: white !important;
}

/* Custom header styling */
.big-title {
    font-size: 2.8rem !important;  /* Larger title */
    font-weight: 800 !important;
    text-align: center;
    margin-bottom: 1.5rem;
    background: linear-gradient(to right, #4e73df, #224abe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
}

.team-name {
    font-size: 1.8rem !important;  /* Larger team name */
    color: var(--text-light) !important;
    text-align: center;
    margin-bottom: 2.5rem;
    font-weight: 600;
}

/* Expander styling */
.streamlit-expanderHeader {
    background-color: var(--bg-light);
    border-radius: 0.35rem;
    padding: 0.75rem 1.25rem;
    margin-bottom: 0.5rem;
    border: none;
    font-weight: 600;
    font-size: 1.1rem !important;  /* Larger expander headers */
}

.streamlit-expanderContent {
    padding: 1rem 1.25rem;
    font-size: 1.05rem !important;  /* Larger content */
}

/* Dataframe styling */
.stDataFrame {
    border-radius: 0.35rem;
    box-shadow: 0 0.15rem 1.75rem 0 rgba(0, 0, 0, 0.2);
    font-size: 1rem !important;  /* Slightly larger table text */
}

/* Alert boxes */
.stAlert {
    border-radius: 0.35rem;
    font-size: 1.05rem !important;  /* Larger alert text */
}

/* Make sure tooltip text is readable */
.stTooltip {
    color: white !important;
    font-size: 1rem !important;
}

/* Card styling for projects */
.card {
    border: none;
    border-radius: 0.35rem;
    box-shadow: 0 0.15rem 1.75rem 0 rgba(0, 0, 0, 0.2);
    transition: transform 0.3s;
    background: var(--bg-light);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    color: var(--text-light) !important;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 0.5rem 1.5rem 0 rgba(0, 0, 0, 0.3);
}

/* Custom divider */
.divider {
    height: 0.2rem;
    background: linear-gradient(to right, var(--primary), var(--success));
    margin: 1.5rem 0;
    border-radius: 0.1rem;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    body {
        font-size: 1rem;
    }
    
    .big-title {
        font-size: 2.2rem !important;
    }
    
    .team-name {
        font-size: 1.5rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

# Update your team name and title section to use the new styling
st.markdown('<div class="big-title">Rudra Pratap Singh Kanawat</div>', unsafe_allow_html=True)
st.markdown('<div class="team-name">Team 51</div>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Update your tabs styling (around line 160)
tabs = st.tabs([
    "🏠 Home", 
    "🖥 Remote Linux/Docker", 
    "🤖 Automation Tools", 
    "🧠 AI Assistants", 
    "☁️ Cloud", 
    "🛠 HTML Tools", 
    "📊 Other Projects", 
    "🔬 ML & NLP Projects",
    "🚀 DevOps Projects"
])
# ------------------ HOME CONTENT ------------------
with tabs[0]:
    st.markdown("""
    ## Welcome to the *Ultimate Multi-Tool Dashboard*!
    
    This comprehensive application combines multiple powerful tools into one convenient interface:
    
    ### 🔌 Remote Management
    - **SSH connection** to Linux servers
    - **Docker container management**
    
    ### 🤖 Automation Suite
    - **WhatsApp, Email, and SMS** automation
    - **Social media posting** (Twitter, Instagram)
    - **System monitoring** and diagnostics
    - **Google Search** integration
    - **Digital Image Creator**
    - **Website scraping** capabilities
    
    ### 🧠 AI Assistants
    - **Legal advice generator**
    - **DevOps career mentor**
    - **Life coaching advisor**
    
    ### ☁️ Cloud Operations
    - **AWS EC2 instance management**
    - **Hand gesture controlled** cloud operations
    - Instance monitoring and control
    
    ### 🛠 Web Tools
    - **Camera photo capture**
    - **Location services**
    - Custom HTML integrations
    
    ### 📊 Data Projects
    - **Ride fare comparator** (Rapido, Ola, Uber)
    - **Stock price predictor** (Linear Regression)
    
    ### 🔬 ML & NLP Projects
    - **Missing Value Predictor** (Linear Regression)
    - **Sentiment Analysis** (Alexa Reviews)
    - **Titanic Survival Predictor**
    
    ---
    
    ### Getting Started
    1. **Select a tab** from the top navigation to explore different functionalities.
    2. **Follow the instructions** within each tool's section.
    3. **Listen for voice feedback** for interactive guidance (requires `gTTS` and `pygobject` or system audio playback).
    
    *Note: Some features require API keys or credentials for full functionality. Please ensure necessary files like datasets are in the same directory.*
    """)

# ------------------ REMOTE LINUX/DOCKER ------------------
with tabs[1]:
    st.header("🔌 Remote Linux & Docker Management")
    
    with st.expander("🖥 SSH Connection Setup", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            ip = st.text_input("IP Address", help="e.g., 192.168.1.100")
        with col2:
            username = st.text_input("Username", help="e.g., root or ubuntu")
        password = st.text_input("Password", type="password", help="Use a strong password or consider SSH keys for production")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Test Ping"):
                if ping_host(ip):
                    st.success("Host is reachable. ✅")
                    speak("Host is reachable")
                else:
                    st.error("Host unreachable. ❌")
                    speak("Host is unreachable")
        with col2:
            if st.button("Connect via SSH"):
                if ip and username and password:
                    if ssh_connect(ip, username, password):
                        speak("SSH connected successfully")
                    else:
                        speak("SSH connection failed")
                else:
                    st.warning("Please fill in all connection details.")
                    speak("Please fill all connection details")

    if "ssh" in st.session_state and st.session_state["ssh"] is not None:
        ssh = st.session_state["ssh"]
        
        tab1, tab2 = st.tabs(["Linux Commands", "Docker Management"])
        
        with tab1:
            st.subheader("🖥 Linux Command Center")
            linux_commands = {
                "System Info": "uname -a",
                "Date": "date",
                "Calendar": "cal",
                "Uptime": "uptime",
                "Memory Usage": "free -h",
                "Disk Usage": "df -h",
                "List Files": "ls -la",
                "Current User": "whoami",
                "Processes": "ps aux",
                "Network Info": "ip addr show || ifconfig", # Use ip addr for modern systems, fallback to ifconfig
                "View /etc/passwd": "cat /etc/passwd",
                "View /etc/shadow (requires root/sudo)": "sudo cat /etc/shadow",
                "View /etc/group": "cat /etc/group",
                "Real-time System Monitor (top -b -n 1)": "top -b -n 1"
            }
            
            col1, col2 = st.columns([1, 3])
            with col1:
                selected_linux_cmd = st.selectbox("Choose Command", list(linux_commands.keys()))
                if st.button("Run Selected Linux Command"):
                    with st.spinner(f"Running '{selected_linux_cmd}'..."):
                        out, err = run_command(ssh, linux_commands[selected_linux_cmd])
                        if out:
                            st.session_state.linux_output = out
                            st.success("Command executed successfully. ✅")
                        elif err:
                            st.session_state.linux_output = err
                            st.error("Command failed. ❌")
                        else:
                            st.session_state.linux_output = "No output or error from command."
                            st.info("Command executed, no output.")
            with col2:
                if 'linux_output' in st.session_state:
                    st.code(st.session_state.linux_output, language="bash")
            
            st.markdown("---")
            st.subheader("Custom Linux Command")
            custom_cmd_input = st.text_input("Enter custom Linux command:")
            if st.button("Run Custom Linux Command"):
                if custom_cmd_input:
                    with st.spinner(f"Running custom command: '{custom_cmd_input}'..."):
                        out, err = run_command(ssh, custom_cmd_input)
                        if out:
                            st.code(out, language="bash")
                            st.success("Custom command executed. ✅")
                        elif err:
                            st.error(f"Error: {err} ❌")
                        else:
                            st.info("Custom command executed, no output.")
                else:
                    st.warning("Please enter a custom command.")
        
        with tab2:
            st.subheader("🐳 Docker Container Management")
            
            docker_cmds = {
                "Docker Version": "docker --version",
                "Docker Info": "docker info",
                "List All Containers": "docker ps -a",
                "List Running Containers": "docker ps",
                "List Images": "docker images",
                "Remove All Stopped Containers": "docker rm $(docker ps -a -q)",
                "Remove All Images (DANGER!)": "docker rmi $(docker images -q) -f", # Added -f for force removal
                "Docker System Info": "docker system info",
                "Docker Disk Usage": "docker system df"
            }
            
            col1, col2 = st.columns([1, 3])
            with col1:
                selected_docker_cmd = st.selectbox("Choose Docker Command", list(docker_cmds.keys()))
                if st.button("Run Selected Docker Command"):
                    with st.spinner(f"Running '{selected_docker_cmd}'..."):
                        out, err = run_command(ssh, docker_cmds[selected_docker_cmd])
                        if out:
                            st.session_state.docker_output = out
                            st.success("Command executed. ✅")
                        elif err:
                            st.session_state.docker_output = err
                            st.error("Command failed. ❌")
                        else:
                            st.session_state.docker_output = "No output or error from command."
                            st.info("Command executed, no output.")
            with col2:
                if 'docker_output' in st.session_state:
                    st.code(st.session_state.docker_output, language="bash")
            
            st.markdown("---")
            st.subheader("Container Operations")
            col1_docker_ops, col2_docker_ops = st.columns(2)
            with col1_docker_ops:
                image = st.text_input("Image to run (e.g. `ubuntu`)")
                if st.button("Run New Container"):
                    if image:
                        with st.spinner(f"Running container from {image}..."):
                            out, err = run_command(ssh, f"docker run -dit {image}")
                            st.code(out if out else err)
                            if not err: st.success("Container started. ✅")
                    else:
                        st.warning("Enter image name.")

                pull_image = st.text_input("Image to pull (e.g. `nginx`)")
                if st.button("Pull Image"):
                    if pull_image:
                        with st.spinner(f"Pulling image {pull_image}..."):
                            out, err = run_command(ssh, f"docker pull {pull_image}")
                            st.code(out if out else err)
                            if not err: st.success("Image pulled. ✅")
                    else:
                        st.warning("Enter image name.")

            with col2_docker_ops:
                cid = st.text_input("Container ID/Name for actions")
                if st.button("Start Container"):
                    if cid:
                        out, err = run_command(ssh, f"docker start {cid}")
                        st.code(out if out else err)
                        if not err: st.success("Container started. ✅")
                if st.button("Stop Container"):
                    if cid:
                        out, err = run_command(ssh, f"docker stop {cid}")
                        st.code(out if out else err)
                        if not err: st.success("Container stopped. ✅")
                if st.button("Remove Container"):
                    if cid:
                        out, err = run_command(ssh, f"docker rm {cid}")
                        st.code(out if out else err)
                        if not err: st.success("Container removed. ✅")
    else:
        st.info("Please connect via SSH to enable remote Linux and Docker commands.")


# ------------------ AUTOMATION TOOLS ------------------
with tabs[2]:
    st.header("🤖 Automation Tools Suite")
    
    tool = st.selectbox("Choose Automation Tool", [
        "WhatsApp", "Email", "SMS", "Phone Call", "Twitter", "Instagram",
        "System Info", "Google Search", "Digital Image Creator", "Website Scraper"
    ])

    if tool == "WhatsApp":
        st.subheader("📲 WhatsApp Automation")
        st.info("This feature uses `pywhatkit` and requires an open browser window for WhatsApp Web. Ensure you are logged into WhatsApp Web on your default browser.")
        with st.form("whatsapp_form"):
            number = st.text_input("Recipient Phone Number (with country code, e.g., +919876543210)")
            msg = st.text_area("Message to send")
            repeat_text = st.text_input("Repeat message (e.g., 'one', '5')", value="1")
            
            if st.form_submit_button("Send WhatsApp"):
                if number and msg:
                    repeat = extract_number(repeat_text)
                    try:
                        speak("Opening WhatsApp Web")
                        st.info("Opening WhatsApp Web... Please wait for a few seconds.")
                        # pywhatkit.sendwhatmsg_instantly opens browser and waits
                        pywhatkit.sendwhatmsg_instantly(number, "Initial check message", wait_time=15, tab_close=False)
                        st.success("WhatsApp Web should be open. Now sending messages.")
                        time.sleep(5) # Give it a moment to stabilize before typing

                        # Use pyautogui to type and send messages
                        for i in range(repeat):
                            pyautogui.write(msg)
                            pyautogui.press("enter")
                            time.sleep(0.5) # Small delay between messages
                        st.success(f"✅ Sent {repeat} time(s).")
                        speak("Message sent successfully")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                        st.warning("Ensure WhatsApp Web is not already open, you are logged in, and your internet connection is stable. Also, ensure your browser is minimized or not obscured.")
                        speak("Failed to send WhatsApp message")
                else:
                    st.warning("Please fill all fields.")

    elif tool == "Email":
        st.subheader("📧 Email Automation")
        st.info("Requires a Gmail account with 2-Factor Authentication enabled and an App Password. [How to generate an App Password](https://support.google.com/accounts/answer/185833?hl=en)")
        with st.form("email_form"):
            sender = st.text_input("Your Gmail Address")
            app_pass = st.text_input("Your Gmail App Password", type="password")
            to = st.text_input("Recipient Email Address")
            subject = st.text_input("Email Subject")
            body = st.text_area("Email Body")
            
            if st.form_submit_button("Send Email"):
                if sender and app_pass and to:
                    try:
                        msg = MIMEMultipart()
                        msg["From"] = sender
                        msg["To"] = to
                        msg["Subject"] = subject
                        msg.attach(MIMEText(body, 'plain'))
                        
                        with st.spinner("Sending email..."):
                            server = smtplib.SMTP('smtp.gmail.com', 587)
                            server.starttls()
                            server.login(sender, app_pass)
                            server.sendmail(sender, to, msg.as_string())
                            server.quit()
                        st.success("✅ Email sent successfully!")
                        speak("Email sent")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                        st.warning("Please double-check your email, App Password, and recipient address. Ensure less secure app access is off (use App Password instead).")
                        speak("Email sending failed")
                else:
                    st.warning("Fill all required fields.")

    elif tool == "SMS":
        st.subheader("📩 SMS Automation")
        st.info("Requires a Twilio account, SID, Auth Token, and a Twilio phone number.")
        with st.form("sms_form"):
            sid = st.text_input("Twilio SID", value="ACdb8e70d6a804bd206583facf2a4fba1d")
            token = st.text_input("Auth Token", type="password", value="b2da3a7d63ba098d0a3f33501b2eb8c4")
            from_num = st.text_input("Your Twilio Phone Number (e.g., +19342465942)", value="+19342465942")
            to_num = st.text_input("Recipient Phone Number (with country code, e.g., +919876543210)", value="+91")
            msg = st.text_input("Message to send")
            
            if st.form_submit_button("Send SMS"):
                if sid and token and from_num and to_num and msg:
                    try:
                        with st.spinner("Sending SMS..."):
                            client = Client(sid, token)
                            message = client.messages.create(body=msg, from_=from_num, to=to_num)
                            st.success(f"✅ SMS sent! Message SID: {message.sid}")
                            speak("SMS sent successfully")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                        st.warning("Please verify your Twilio credentials and phone numbers. Ensure sufficient balance.")
                        speak("SMS sending failed")
                else:
                    st.warning("Please fill all fields.")

    elif tool == "Phone Call":
        st.subheader("📞 Phone Call Automation")
        st.info("Requires a Twilio account and a TwiML Bin URL for call instructions.")
        with st.form("call_form"):
            sid = st.text_input("Twilio SID", key="call_sid", value="ACf22a58c4ec38d6cc585a0764c95041fc")
            token = st.text_input("Auth Token", type="password", key="call_token", value="f97bcf7056fca819e2eeda9bae2737b7")
            from_num = st.text_input("Your Twilio Phone Number", key="call_from_num", value="+14632707601")
            to_num = st.text_input("Recipient Phone Number", key="call_to_num", value="+91")
            twiml_url = st.text_input("TwiML Bin URL",
                                      value="https://handler.twilio.com/twiml/EHXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
                                      help="This URL provides instructions (e.g., text-to-speech) for the call.")
            
            if st.form_submit_button("Make Call"):
                if sid and token and from_num and to_num and twiml_url:
                    try:
                        with st.spinner("Initiating call..."):
                            client = Client(sid, token)
                            call = client.calls.create(to=to_num, from_=from_num, url=twiml_url)
                            st.success(f"📞 Call initiated! SID: {call.sid}")
                            speak("Call made successfully")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                        st.warning("Please verify your Twilio credentials, phone numbers, and TwiML URL. Ensure sufficient balance.")
                        speak("Call failed")
                else:
                    st.warning("Fill all fields.")

    elif tool == "Twitter":
        st.subheader("🐦 Twitter Automation")
        st.info("Make sure your Twitter Developer App has 'Read and Write' permissions.")
        with st.form("twitter_form"):
            tweet_text = st.text_area("What's happening?", max_chars=280)
            uploaded_file = st.file_uploader("Upload image for Tweet", type=["jpg", "jpeg", "png"])
            
            # Twitter API credentials (hardcoded for demo, use st.secrets for production)
            bearer_token = 'AAAAAAAAAAAAAAAAAAAAAPYw3AEAAAAAT84MYzMPrkVwBRSdtf%2BSjjZTcSo%3DGS2v2vweChS5JbwpdblTq1lHEWjEynakxZlYxG4KAYUFw13mAY'
            consumer_key = 'HRTMFeBVGzMH00FAYkdxwDFJ8'
            consumer_secret = 'QTw8uI60ZDEIXsOwtmMCVF3w5UTqx5Tu5KizD84iRnTBjYGvrX'
            access_token = '1941841332697522176-vISAAeu2l7QvQmNcPlLxl0E7LIfEYl'
            access_token_secret = '68juASxBu7wTgyvMORnGREhoNPAL1TLYkkRsn1jB1yVHK'

            if st.form_submit_button("Tweet Now"):
                if tweet_text and uploaded_file:
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                            tmp.write(uploaded_file.read())
                            image_path = tmp.name

                        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
                        auth.set_access_token(access_token, access_token_secret)
                        api = tweepy.API(auth, wait_on_rate_limit=True)

                        client_v2 = tweepy.Client(
                            bearer_token=bearer_token,
                            consumer_key=consumer_key,
                            consumer_secret=consumer_secret,
                            access_token=access_token,
                            access_token_secret=access_token_secret,
                            wait_on_rate_limit=True
                        )

                        media = api.media_upload(filename=image_path)
                        media_id = media.media_id_string
                        response = client_v2.create_tweet(text=tweet_text, media_ids=[media_id])

                        st.success("✅ Tweet posted!")
                        st.json(response.data)
                        speak("Tweet posted successfully")
                        os.remove(image_path) # Clean up temp file
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                        st.warning("Ensure your API keys are correct and your app has the necessary permissions. Also check internet connection.")
                        speak("Tweet failed")
                else:
                    st.warning("Upload image and enter text.")

    elif tool == "Instagram":
        st.subheader("📸 Instagram Automation")
        st.info("`instagrapi` can sometimes require a re-login. Avoid frequent use to prevent account flags.")
        with st.form("instagram_form"):
            username = st.text_input("Instagram Username")
            password = st.text_input("Instagram Password", type="password")
            caption = st.text_input("Image Caption")
            uploaded_file = st.file_uploader("Upload an image for Instagram", type=["jpg", "jpeg"])
            
            if st.form_submit_button("Upload to Instagram"):
                if not username or not password or not uploaded_file:
                    st.warning("⚠️ Please provide all inputs.")
                else:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.read())

                        try:
                            cl = InstaClient()
                            cl.login(username, password)
                            cl.photo_upload(temp_path, caption)
                            st.success("✅ Successfully uploaded to Instagram!")
                            speak("Image uploaded to Instagram")
                        except Exception as e:
                            st.error(f"❌ Upload failed: {e}")
                            st.warning("Double-check your Instagram credentials. If 2FA is enabled, you might need to approve the login attempt, or use an app-specific password.")
                            speak("Instagram upload failed")

    # --- SYSTEM INFO SECTION (MOVED HERE) ---
    elif tool == "System Info":
        st.subheader("💻 System Resource Monitor")
        st.markdown("View real-time system resource usage and detailed information.")

        # Get system information
        ram_info = get_ram_info()
        cpu_usage = psutil.cpu_percent(interval=1) # Get CPU usage over 1 second

        # Determine the correct root path based on the operating system
        if os.name == 'nt':  # 'nt' is for Windows
            root_path = os.getenv('SystemDrive', 'C:') + '\\'
        else:  # For Unix-like systems (Linux, macOS)
            root_path = '/'

        try:
            disk_usage = psutil.disk_usage(root_path)
        except Exception as e:
            st.error(f"Could not retrieve disk usage for '{root_path}': {e}. This may happen on some systems or virtual environments.")
            disk_usage = None # Handle case where disk_usage cannot be retrieved

        boot_time = psutil.boot_time()

        # Display metrics in columns
        col_sys1, col_sys2, col_sys3 = st.columns(3)

        with col_sys1:
            st.metric("Total RAM", f"{ram_info['total']:.2f} GB")
            st.metric("Available RAM", f"{ram_info['available']:.2f} GB")
            st.progress(ram_info['usage'] / 100)
            st.caption(f"RAM Usage: {ram_info['usage']:.1f}%")

        with col_sys2:
            st.metric("CPU Usage", f"{cpu_usage}%")
            st.progress(cpu_usage / 100)
            st.metric("System Uptime", time.strftime("%H:%M:%S", time.gmtime(time.time() - boot_time)))

        with col_sys3:
            if disk_usage:
                st.metric("Total Disk Space", f"{disk_usage.total / (1024**3):.2f} GB")
                st.metric("Used Disk Space", f"{disk_usage.used / (1024**3):.2f} GB")
                st.progress(disk_usage.percent / 100)
                st.caption(f"Disk Usage: {disk_usage.percent:.1f}%")
            else:
                st.info("Disk usage data not available.")

        with st.expander("Detailed System Information"):
            st.subheader("CPU Information")
            st.write(f"**Physical Cores:** {psutil.cpu_count(logical=False)}")
            st.write(f"**Logical Cores:** {psutil.cpu_count(logical=True)}")
            st.write(f"**Current Frequency:** {psutil.cpu_freq().current:.2f} MHz")

            st.subheader("Network Information")
            net_io = psutil.net_io_counters()
            st.write(f"**Bytes Sent:** {net_io.bytes_sent / (1024**2):.2f} MB")
            st.write(f"**Bytes Received:** {net_io.bytes_recv / (1024**2):.2f} MB")

            st.subheader("Running Processes (Top 20)")
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 'memory_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
            df_processes = pd.DataFrame(processes).sort_values(by='cpu_percent', ascending=False)
            st.dataframe(df_processes.head(20))

    elif tool == "Google Search":
        st.subheader("🔍 Google Search Automation")
        st.info("This tool performs a Google search and displays the top results.")
        with st.form("search_form"):
            query = st.text_input("Search Query", "best Python libraries for web scraping")
            num_results = st.slider("Number of Results", 1, 20, 10)
            
            if st.form_submit_button("Search Google"):
                try:
                    st.subheader(f"Results for: '{query}'")
                    
                    results = []
                    for i, result in enumerate(search(query, num_results=num_results)):
                        results.append(result)
                        st.write(f"{i+1}. {result}")
                        
                        # Attempt to fetch title and snippet for better display
                        try:
                            resp = requests.get(result, timeout=3) # Shorter timeout for snippets
                            soup = BeautifulSoup(resp.text, "html.parser")
                            title = soup.title.string.strip() if soup.title else "No Title Available"
                            snippet_tag = soup.find("meta", attrs={"name": "description"})
                            snippet = snippet_tag["content"].strip() if snippet_tag else "No Description Snippet Available."
                            st.markdown(f"**Title:** {title}")
                            st.caption(snippet)
                        except Exception as e_fetch:
                            st.caption(f"Could not fetch snippet: {e_fetch}")
                        st.markdown("---")

                    if results:
                        df = pd.DataFrame(results, columns=["URL"])
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download Results as CSV",
                            data=csv,
                            file_name="Google Search_results.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No results found for your query.")
                except Exception as e:
                    st.error(f"Error performing search: {e}")
                    st.info("Make sure 'google' package is installed (`pip install google`). Network issues or rate limits can also cause this.")

    elif tool == "Digital Image Creator":
        st.subheader("🎨 Digital Image Creator")
        st.info("Create simple digital images with custom shapes, text, and colors.")
        with st.form("image_creator_form"):
            col1, col2 = st.columns(2)
            with col1:
                width = st.slider("Image Width", 100, 1200, 800)
                height = st.slider("Image Height", 100, 1200, 400)
                bg_color = st.color_picker("Background Color", "#1E90FF")  # DodgerBlue
            with col2:
                rect_color = st.color_picker("Rectangle Outline Color", "#FFFFFF")
                circle_color = st.color_picker("Circle Fill Color", "#FFFF00")
                text_color = st.color_picker("Text Color", "#FFFFFF")
            
            custom_text = st.text_input("Text to Display", "Hello from Python!")
            
            if st.form_submit_button("Generate Image"):
                try:
                    image = Image.new("RGB", (width, height), color=hex_to_rgb(bg_color))
                    draw = ImageDraw.Draw(image)
                    
                    # Draw shapes
                    draw.rectangle([(50, 50), (width-50, height-50)], outline=hex_to_rgb(rect_color), width=4)
                    
                    circle_size = min(width, height) // 8
                    draw.ellipse([
                        (width - circle_size - 100, 100),
                        (width - 100, 100 + circle_size)
                    ], fill=hex_to_rgb(circle_color), outline="black")
                    
                    # Add text (centered)
                    # Note: ImageFont.load_default() is basic, for better fonts, need .ttf files
                    try:
                        font = ImageFont.truetype("arial.ttf", 30) # Try a common font, adjust size
                    except IOError:
                        font = ImageFont.load_default()
                        st.warning("Arial font not found, using default font. For better font options, install 'arial.ttf' or similar.")

                    text_bbox = draw.textbbox((0,0), custom_text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]

                    text_x = (width - text_width) // 2
                    text_y = (height - text_height) // 2 - 10 # Adjust slightly up

                    draw.text(
                        (text_x, text_y),
                        custom_text,
                        font=font,
                        fill=hex_to_rgb(text_color)
                    )
                    
                    st.image(image, caption="Your Generated Image", use_column_width=True)
                    
                    # Save and offer download
                    image_byte_arr = BytesIO()
                    image.save(image_byte_arr, format="PNG")
                    image_byte_arr.seek(0)
                    
                    st.download_button(
                        "Download Image",
                        data=image_byte_arr,
                        file_name="generated_image.png",
                        mime="image/png"
                    )
                    
                except Exception as e:
                    st.error(f"Error generating image: {e}")
                    st.info("Make sure 'Pillow' is installed: `pip install Pillow`")

    elif tool == "Website Scraper":
        st.subheader("🕸️ Website Data Downloader")
        st.info("Crawl and download text content and links from a specified website. Be mindful of website terms of service and robots.txt.")

        url_to_scrape = st.text_input("Enter website URL to scrape:", "https://example.com")
        max_pages_scrape = st.number_input("Maximum pages to scrape (within domain):", min_value=1, max_value=50, value=5)
        domain_only_scrape = st.checkbox("Only scrape pages from the same domain (recommended)", value=True)
        download_format_scrape = st.selectbox("Select download format:", ["csv", "json", "excel"])

        if st.button("Start Scraping and Download Data", type="primary"):
            if not url_to_scrape:
                st.error("Please enter a URL to start scraping.")
            elif not is_valid_url(url_to_scrape):
                st.error("Please enter a valid URL (e.g., `http://example.com` or `https://example.com`).")
            else:
                with st.spinner(f"Initiating scraping for {url_to_scrape}... This may take a moment depending on the number of pages."):
                    scraped_data = scrape_website_data(url_to_scrape, max_pages_scrape, domain_only_scrape)

                if scraped_data:
                    st.success(f"Successfully scraped **{len(scraped_data)}** pages!")
                    download_bytes = save_data(scraped_data, download_format_scrape)
                    st.download_button(
                        label=f"Download Scraped Data as .{download_format_scrape.upper()}",
                        data=download_bytes,
                        file_name=f"scraped_website_data.{download_format_scrape}",
                        mime={
                            'csv': 'text/csv',
                            'json': 'application/json',
                            'excel': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                        }[download_format_scrape]
                    )

                    st.subheader("Preview of Scraped Data (First 5 entries)")
                    st.dataframe(pd.DataFrame(scraped_data).head())
                else:
                    st.error("No data was scraped. This could be due to network issues, website blocks, or no accessible links.")

        st.warning("""
        **Important Notes on Web Scraping:**
        1. Always check the website's `robots.txt` file and terms of service before scraping.
        2. This is a basic scraper - some websites may require more advanced techniques (e.g., handling JavaScript, CAPTCHAs).
        3. Be respectful with scraping frequency to avoid overwhelming servers and getting your IP blocked.
        """)


# ------------------ AI ASSISTANTS ------------------
with tabs[3]:
    # Initialize Gemini model
    model = setup_gemini()
    
    st.header("🧠 AI Assistants")
    st.markdown("""
    Multiple AI assistants powered by **Google Gemini**:
    - **🧑‍⚖ Legal Assistant**: Explains legal queries in simple terms.
    - **🛠 DevOps Career Mentor**: Provides advice on DevOps tools, certifications, and job roles.
    - **🧠 Life Advisor**: Offers practical guidance for personal and professional challenges.
    
    > ⚠ **Disclaimer**: These assistants provide general information and are not substitutes for professional legal, career, or personal advice. Always consult a qualified professional for specific situations.
    """)
    
    # Create tabs for different AI assistants
    tab_ai_1, tab_ai_2, tab_ai_3 = st.tabs(["🧑‍⚖ Legal Assistant", "🛠 DevOps Career Mentor", "🧠 Life Advisor"])

    # Legal Assistant Tab
    with tab_ai_1:
        st.subheader("Legal Assistant")
        with st.form("legal_form"):
            legal_q = st.text_area("Your legal question", placeholder="e.g., What are the basic terms of a non-disclosure agreement?", height=150)
            if st.form_submit_button("Get Legal Help"):
                if not legal_q.strip():
                    st.warning("Please enter a legal question to get started.")
                else:
                    with st.spinner("Analyzing your legal question..."):
                        try:
                            prompt = f"As a legal expert, explain this question in simple terms, citing general principles, and suggest possible general legal steps or considerations (do not provide specific legal advice or endorse any action):\n{legal_q}"
                            response = model.generate_content(prompt)
                            st.text_area("Legal Advice", value=response.text, height=300, disabled=True)
                        except Exception as e:
                            st.error(f"Error generating legal advice: {str(e)}")
                            st.info("The AI model might be experiencing issues or the query was too complex. Please try again or rephrase.")

    # DevOps Career Mentor Tab
    with tab_ai_2:
        st.subheader("DevOps Career Mentor")
        
        # Initialize chat session if not exists
        if "devops_chat" not in st.session_state:
            st.session_state.devops_chat = model.start_chat(
                history=[{"role": "user", "parts": ["You are a friendly and knowledgeable DevOps career mentor. You guide individuals about various DevOps tools, certifications, job roles, and career paths in the IT industry. Keep your responses encouraging and informative."]},
                         {"role": "model", "parts": ["Hello! I'm your DevOps career mentor. What specific questions do you have about DevOps, its tools, certifications, or career opportunities? I'm here to help you navigate this exciting field!"]}]
            )

        # Display chat messages from history
        for message in st.session_state.devops_chat.history[1:]: # Skip initial setup message
            role = "🧑‍💻 You" if message.role == "user" else "🧠 Mentor"
            st.markdown(f"**{role}:** {message.parts[0].text}")

        with st.form("devops_form"):
            devops_q = st.text_area("Your DevOps question", placeholder="e.g., What are the essential tools for a beginner in DevOps? Or, What certifications are highly valued?", height=150)
            if st.form_submit_button("Get DevOps Advice"):
                if not devops_q.strip():
                    st.warning("Please enter your DevOps question.")
                else:
                    with st.spinner("Consulting DevOps expert..."):
                        try:
                            reply = st.session_state.devops_chat.send_message(devops_q)
                            # The message is added to history and displayed by the loop above
                        except Exception as e:
                            st.error(f"Error getting DevOps advice: {str(e)}")
                            st.info("The AI model might be experiencing issues. Please try again or rephrase your question.")

    # Life Advisor Tab
    with tab_ai_3:
        st.subheader("Life Advisor")
        st.markdown("Get practical and empathetic advice for personal and professional challenges. Remember, this is AI-generated advice and not a substitute for professional counseling.")
        
        # Initialize life advisor chat if not exists
        if "life_chat" not in st.session_state:
            st.session_state.life_chat = model.start_chat(
                history=[{"role": "user", "parts": ["You are a helpful and empathetic life advisor. Provide practical, kind, and non-judgmental advice for personal and professional challenges. Focus on actionable steps and general well-being, always reminding the user that this is AI advice and not professional counseling."]},
                         {"role": "model", "parts": ["Hi there! I'm here to help you think through life's challenges. What's on your mind today?"]}]
            )

        for message in st.session_state.life_chat.history[1:]: # Skip initial setup message
            role = "🧑‍💻 You" if message.role == "user" else "🧠 Advisor"
            st.markdown(f"**{role}:** {message.parts[0].text}")

        with st.form("life_form"):
            life_q = st.text_area(
                "Describe your life challenge or question",
                placeholder="e.g., I'm feeling overwhelmed with work-life balance. Any tips?",
                height=150
            )
            
            if st.form_submit_button("Get Life Advice"):
                if not life_q.strip():
                    st.warning("Please describe your challenge to get advice.")
                else:
                    with st.spinner("Thinking about your situation..."):
                        try:
                            prompt = f"Give practical, kind advice for this real-life situation, remembering to emphasize that this is AI advice:\n{life_q}"
                            response = st.session_state.life_chat.send_message(prompt)
                            # The message is added to history and displayed by the loop above
                        except Exception as e:
                            st.error(f"Error getting life advice: {str(e)}")
                            st.info("The AI model might be experiencing issues. Please try again or rephrase your question.")


# ------------------ CLOUD OPERATIONS ------------------
with tabs[4]:
    st.header("☁️ Cloud Operations Dashboard")
    
    cloud_option = st.selectbox("Select Cloud Operation",
                                ["AWS EC2 Hand Gesture Control", "Manual EC2 Management"])
    
    if cloud_option == "AWS EC2 Hand Gesture Control":
        st.subheader("🤖 Hand Gesture Controlled EC2 Instance")
        st.info("Show **all 5 fingers** to the camera to enable EC2 instance launch options. Your hand should be clearly visible.")
        
        # Initialize session state for hand gesture data with proper defaults
        if 'hand_gesture_data' not in st.session_state:
            st.session_state.hand_gesture_data = {
                'photo_captured': None,
                'fingers_detected': None,
                'camera_index': 0
            }
        
        with st.form("gesture_form"):
            st.session_state.hand_gesture_data['camera_index'] = st.number_input(
                "Camera Index (usually 0 for built-in webcam)",
                min_value=0,
                max_value=10,
                value=st.session_state.hand_gesture_data['camera_index']
            )
            
            if st.form_submit_button("Capture Photo for Gesture Detection"):
                cap = None
                try:
                    cap = cv2.VideoCapture(st.session_state.hand_gesture_data['camera_index'])
                    
                    if not cap.isOpened():
                        st.error("❌ Could not open camera. Try a different camera index or ensure camera drivers are updated.")
                    else:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        
                        time.sleep(2.0) # Allow camera to warm up
                        
                        ret, frame = cap.read() # Capture a single frame
                            
                        if ret:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            
                            detector = HandDetector(maxHands=1, detectionCon=0.7)
                            hands, img_with_hands = detector.findHands(frame_rgb, draw=True)
                            
                            if hands:
                                hand = hands[0]
                                fingers = detector.fingersUp(hand)
                                st.session_state.hand_gesture_data['fingers_detected'] = fingers
                                
                                cv2.putText(img_with_hands, f"Fingers Up: {fingers.count(1)}", (50, 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                                
                                st.session_state.hand_gesture_data['photo_captured'] = img_with_hands
                                
                                if fingers == [1, 1, 1, 1, 1]:
                                    st.success("🎉 All 5 fingers detected! AWS EC2 launch options are now available below.")
                                else:
                                    st.warning(f"Detected {fingers.count(1)} fingers. Show all 5 fingers to enable EC2 launch.")
                            else:
                                st.warning("🤚 No hands detected in the photo. Please ensure your hand is visible.")
                                st.session_state.hand_gesture_data['photo_captured'] = frame_rgb
                        else:
                            st.error("📷 Failed to capture photo from camera. Retrying might help.")
                            
                except Exception as e:
                    st.error(f"⚠️ An error occurred with camera/gesture detection: {str(e)}")
                finally:
                    if cap is not None:
                        cap.release()
            
        # Display captured photo if available
        if st.session_state.hand_gesture_data['photo_captured'] is not None:
            try:
                st.image(
                    st.session_state.hand_gesture_data['photo_captured'],
                    caption="Captured Image (Hand detection overlay)",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error displaying image: {str(e)}")
        else:
            st.info("No image captured yet. Click the 'Capture Photo' button above.")
            
        # If all 5 fingers detected, show AWS credential inputs
        if st.session_state.hand_gesture_data.get('fingers_detected') == [1, 1, 1, 1, 1]:
            st.subheader("🔑 AWS Credentials to Launch EC2")
            with st.form("aws_credentials_form"):
                aws_access = st.text_input("AWS Access Key ID", type="password")
                aws_secret = st.text_input("AWS Secret Access Key", type="password")
                region = st.selectbox("AWS Region", ["ap-south-1", "us-east-1", "us-west-2", "eu-west-1"])
                
                if st.form_submit_button("🚀 Launch EC2 Instance"):
                    if aws_access and aws_secret:
                        try:
                            ec2_client = boto3.client(
                                'ec2',
                                region_name=region,
                                aws_access_key_id=aws_access,
                                aws_secret_access_key=aws_secret
                            )

                            response = ec2_client.run_instances(
                                ImageId='ami-0d0ad8bb301edb745',
                                InstanceType='t2.micro',
                                MinCount=1,
                                MaxCount=1
                            )
                            
                            instance_id = response['Instances'][0]['InstanceId']
                            st.success(f"✅ EC2 instance launched successfully! Instance ID: **{instance_id}**")
                            
                            # Reset after successful launch
                            st.session_state.hand_gesture_data = {
                                'photo_captured': None,
                                'fingers_detected': None,
                                'camera_index': st.session_state.hand_gesture_data['camera_index']
                            }
                        except Exception as e:
                            st.error(f"❌ Error launching instance: {str(e)}")
                    else:
                        st.warning("⚠️ Please enter both AWS Access Key and Secret Key.")

    elif cloud_option == "Manual EC2 Management":
        st.subheader("🛠 Manual EC2 Instance Management")
        
        with st.form("aws_manual_credentials"):
            aws_access_manual = st.text_input("AWS Access Key ID", key="aws_access_manual", type="password")
            aws_secret_manual = st.text_input("AWS Secret Access Key", key="aws_secret_manual", type="password")
            region_manual = st.selectbox("AWS Region", ["ap-south-1", "us-east-1", "us-west-2", "eu-west-1"], key="region_manual")
            
            if st.form_submit_button("Connect to AWS"):
                if aws_access_manual and aws_secret_manual:
                    try:
                        ec2_resource = boto3.resource(
                            'ec2',
                            region_name=region_manual,
                            aws_access_key_id=aws_access_manual,
                            aws_secret_access_key=aws_secret_manual
                        )
                        st.session_state.ec2_manual = ec2_resource
                        st.success("Successfully connected to AWS. ✅")
                    except Exception as e:
                        st.error(f"AWS Connection Error: {str(e)} ❌")
                        st.warning("Please check your AWS Access Key ID, Secret Access Key, and ensure your account has the necessary permissions.")
                else:
                    st.warning("Please enter both AWS Access Key and Secret Key.")
            
        if 'ec2_manual' in st.session_state and st.session_state.ec2_manual is not None:
            ec2 = st.session_state.ec2_manual
            
            st.markdown("---")
            with st.expander("Launch New Instance", expanded=True):
                with st.form("launch_instance_form_manual"):
                    ami_manual = st.text_input("AMI ID", value="ami-0d0ad8bb301edb745", help="Ensure this AMI is valid for your selected region.")
                    instance_type_manual = st.selectbox("Instance Type", ["t2.micro", "t2.small", "t2.medium"])
                    key_name_manual = st.text_input("Key Pair Name", help="Required for SSH access to the instance.")
                    security_group_manual = st.text_input("Security Group ID", help="e.g., sg-xxxxxxxxxxxxxxxxx. Allows defining network access rules.")
                    
                    if st.form_submit_button("Launch Instance"):
                        if ami_manual and instance_type_manual:
                            try:
                                instances = ec2.create_instances(
                                    ImageId=ami_manual,
                                    InstanceType=instance_type_manual,
                                    KeyName=key_name_manual,
                                    SecurityGroupIds=[security_group_manual] if security_group_manual else [],
                                    MinCount=1,
                                    MaxCount=1
                                )
                                st.success(f"Instance launched with ID: **{instances[0].id}** ✅")
                            except Exception as e:
                                st.error(f"Error launching instance: {str(e)} ❌")
                                st.warning("Check AMI ID, Instance Type, Key Pair Name, and Security Group ID for validity and permissions.")
                        else:
                            st.warning("AMI ID and Instance Type are required.")
            
            st.markdown("---")
            if st.button("List All Instances", type="primary"):
                with st.spinner("Fetching instances..."):
                    instances = ec2.instances.all()
                    instance_data = []
                    
                    for instance in instances:
                        instance_name = "N/A"
                        if instance.tags:
                            for tag in instance.tags:
                                if tag['Key'] == 'Name':
                                    instance_name = tag['Value']
                                    break
                        instance_data.append({
                            "Name": instance_name,
                            "Instance ID": instance.id,
                            "Type": instance.instance_type,
                            "State": instance.state["Name"],
                            "Public IP": instance.public_ip_address,
                            "Launch Time": instance.launch_time.strftime("%Y-%m-%d %H:%M:%S") if instance.launch_time else "N/A"
                        })
                    
                    if instance_data:
                        st.subheader("Your EC2 Instances:")
                        st.dataframe(pd.DataFrame(instance_data))
                    else:
                        st.info("No EC2 instances found in this region for your account.")
                
            st.markdown("---")
            st.subheader("Instance Control (Start, Stop, Terminate)")
            instance_id_control = st.text_input("Enter Instance ID to manage:", help="e.g., i-0abcdef1234567890")
            
            if instance_id_control:
                col1_control, col2_control, col3_control = st.columns(3)
                with col1_control:
                    if st.button("Start Instance"):
                        try:
                            instance = ec2.Instance(instance_id_control)
                            instance.start()
                            st.success(f"Starting instance **{instance_id_control}** ✅")
                        except Exception as e:
                            st.error(f"Error starting instance: {str(e)} ❌")
                            st.warning("Ensure the Instance ID is correct and it's not already running or in a transition state.")
                
                with col2_control:
                    if st.button("Stop Instance"):
                        try:
                            instance = ec2.Instance(instance_id_control)
                            instance.stop()
                            st.success(f"Stopping instance **{instance_id_control}** ✅")
                        except Exception as e:
                            st.error(f"Error stopping instance: {str(e)} ❌")
                            st.warning("Ensure the Instance ID is correct and it's not already stopped or in a transition state.")
                
                with col3_control:
                    if st.button("Terminate Instance"):
                        if st.warning("⚠️ Are you sure you want to terminate this instance? This action cannot be undone."):
                            confirm_terminate = st.checkbox("Yes, I understand and want to terminate.")
                            if confirm_terminate:
                                try:
                                    instance = ec2.Instance(instance_id_control)
                                    instance.terminate()
                                    st.success(f"Terminating instance **{instance_id_control}** ✅")
                                except Exception as e:
                                    st.error(f"Error terminating instance: {str(e)} ❌")
                                    st.warning("Ensure the Instance ID is correct and it's not already terminated.")
                            else:
                                st.info("Termination cancelled.")
            else:
                st.info("Enter an Instance ID to use the control buttons.")


# ------------------ HTML TOOLS ------------------
with tabs[5]:
    st.header("🛠 HTML Tools Dashboard")
    st.info("""
    This interactive dashboard is designed to integrate web-based functionalities directly into your Streamlit app.
    It can host custom HTML, JavaScript, and CSS for a more dynamic experience.
    """)
    
    # Try to load the HTML file
    st.subheader("Default HTML Interface (from `first.html`)")
    try:
        with open("first.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        components.html(html_content, height=600, scrolling=True) # Adjust height as needed
    except FileNotFoundError:
        st.error("`first.html` not found in the current directory.")
        st.info("Using a sample interface instead. Create `first.html` to load your custom content.")
        components.html("""
            <div style="padding:20px;background:#f0f2f6;border-radius:10px; text-align: center;">
                <h3 style="color:#007bff;">Sample HTML Tools Interface</h3>
                <p>This area is where your `first.html` content would appear.</p>
                <p>You can embed interactive forms, custom JavaScript widgets, or even live camera feeds here.</p>
                <button style="padding: 10px 20px; background-color: #28a745; color: white; border: none; border-radius: 5px; cursor: pointer;" onclick="alert('Hello from HTML!')">Click Me!</button>
                <p style="margin-top: 15px; font-size: 0.9em; color: #6c757d;">(This is a placeholder. Upload your HTML below!)</p>
            </div>
        """, height=250)
    
    st.markdown("---")
    st.subheader("Upload Custom HTML File")
    uploaded_file_html = st.file_uploader("Choose an HTML file to display", type=["html"], key="html_uploader_custom")
    if uploaded_file_html:
        try:
            html_content_custom = uploaded_file_html.read().decode("utf-8")
            st.success(f"Successfully loaded `{uploaded_file_html.name}`! Displaying below:")
            components.html(html_content_custom, height=800, scrolling=True) # Display uploaded HTML
        except Exception as e:
            st.error(f"Error loading custom HTML file: {str(e)}")
            st.warning("Please ensure the uploaded file is a valid HTML document.")
            
    st.markdown("""
    **Important Notes about HTML Integrations:**
    - Some HTML features (like camera/geolocation access) may only work when your Streamlit app is served over HTTPS. This generally isn't an issue on `localhost` but can be on public deployments without proper SSL.
    - Browser security policies (like CORS) might restrict JavaScript in your HTML from making requests to external domains without proper server-side configuration.
    - For complex interactive elements, consider using Streamlit's native components or custom components where possible.
    """)

# ------------------ OTHER PROJECTS ------------------
with tabs[6]:
    st.header("📊 Other Projects")
    
    project = st.selectbox("Choose Project", ["Ride Fare Comparator", "Stock Price Predictor"])
    
    if project == "Ride Fare Comparator":
        st.subheader("🚕 Ride Fare Comparator")
        st.write("Get the *cheapest fare* among Rapido, Ola, and Uber based on mock data.")

        with st.form("ride_form"):
            source = st.text_input("Enter Source Location:", placeholder="e.g., Connaught Place")
            destination = st.text_input("Enter Destination Location:", placeholder="e.g., Delhi Airport")
            
            if st.form_submit_button("Compare Prices"):
                if source and destination:
                    st.info(f"Calculating prices from **{source}** to **{destination}**...")
                    
                    prices = get_mock_prices(source, destination)
                    
                    cheapest_service = min(prices, key=prices.get)

                    st.subheader("💰 Price Comparison:")
                    for service, price in prices.items():
                        if service == cheapest_service:
                            st.success(f"**{service}**: ₹{price:.2f} ✅ Cheapest")
                        else:
                            st.write(f"**{service}**: ₹{price:.2f}")

                    st.markdown("---")
                    st.caption("Note: Prices are mock data for demo purposes and do not reflect real-time fares.")
                else:
                    st.warning("Please enter both source and destination to get fare estimates.")
            
    elif project == "Stock Price Predictor":
        st.subheader("📈 Stock Price Prediction using Linear Regression")
        st.info("Predict future closing stock prices based on historical data. This uses a simple Linear Regression model and is for demonstration purposes only. Do not use for actual financial decisions.")
        with st.form("stock_form"):
            stock_symbol = st.text_input("Enter Stock Symbol (e.g., TATAMOTORS.NS for NSE, AAPL for NASDAQ)", "TATAMOTORS.NS")
            start_date = st.date_input("Start Date for Historical Data", pd.to_datetime("2022-01-01"))
            end_date = st.date_input("End Date for Historical Data", pd.to_datetime("2024-07-28")) # Updated to current date
            future_days = st.slider("Days to Predict into the Future", min_value=7, max_value=60, value=30)
            
            if st.form_submit_button("Predict Stock Price"):
                if stock_symbol and start_date < end_date:
                    try:
                        with st.spinner(f"Downloading {stock_symbol} data and predicting..."):
                            stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
                            
                            if stock_data.empty:
                                st.error(f"❌ No data found for {stock_symbol} between {start_date} and {end_date}. Check symbol and date range.")
                                raise ValueError("Empty stock data")

                            stock_data_clean = stock_data[['Close']].dropna()
                            if stock_data_clean.empty:
                                st.error("❌ 'Close' price data is missing or empty for the selected period.")
                                raise ValueError("Empty 'Close' data")

                            # Create a 'Prediction' column shifted by future_days
                            stock_data_clean['Prediction'] = stock_data_clean[['Close']].shift(-future_days)

                            # Prepare data for training
                            X = np.array(stock_data_clean[['Close']][:-future_days])
                            y = np.array(stock_data_clean['Prediction'][:-future_days])
                            
                            if len(X) == 0:
                                st.error("Not enough data to train the model. Try a longer date range or fewer prediction days.")
                                raise ValueError("Insufficient data for training")

                            # Data for future prediction
                            x_future = np.array(stock_data_clean[['Close']][-future_days:])

                            # Split and train
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                            model_stock = LinearRegression()
                            model_stock.fit(X_train, y_train)

                            # Make predictions
                            predictions = model_stock.predict(x_future)

                            # Prepare dates for predicted future
                            last_date = stock_data_clean.index[-1]
                            future_dates = pd.date_range(start=last_date, periods=future_days + 1, freq='B')[1:] # Business days
                            predicted_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': predictions})
                            
                            st.success("Prediction complete! ✅")

                            st.subheader("📊 Predicted Stock Prices:")
                            
                            # Plotting historical and predicted prices
                            fig, ax = plt.subplots(figsize=(12, 6))
                            
                            # Plot last 90 days of historical data
                            ax.plot(stock_data_clean.index[-90:], stock_data_clean['Close'][-90:], label="Actual Historical Price", color='blue')
                            
                            # Plot predicted prices
                            ax.plot(predicted_df['Date'], predicted_df['Predicted Price'], label=f"Predicted Price (Next {future_days} Days)", linestyle='--', color='red', marker='o', markersize=3)
                            
                            ax.set_xlabel("Date")
                            ax.set_ylabel("Stock Price (USD/INR)")
                            ax.set_title(f"Stock Price Prediction for {stock_symbol}")
                            ax.legend()
                            ax.grid(True)
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(fig)

                            # Display predicted prices table
                            st.dataframe(predicted_df.set_index('Date'))

                    except Exception as e:
                        st.error(f"Error during stock prediction: {str(e)} ❌")
                        st.warning("Possible issues: Incorrect stock symbol, no data for the date range, or network problems. Try a common symbol like 'AAPL' or 'GOOG'.")
                else:
                    st.warning("Please enter a valid stock symbol and ensure end date is after start date.")

# ------------------ ML & NLP PROJECTS (NEW TAB) ------------------
with tabs[7]:
    st.header("🔬 Machine Learning & Natural Language Processing Projects")
    st.markdown("Explore various ML and NLP models for different tasks.")
    
    ml_nlp_project = st.selectbox("Choose ML/NLP Project", [
        "Missing Value Predictor", "Sentiment Analysis", "Titanic Survival Predictor"
    ])

    if ml_nlp_project == "Missing Value Predictor":
        st.subheader("📊 Fill Missing 'Y' Values using Linear Regression")
        st.info("Upload a CSV file and predict missing values in a designated 'Y' column using Linear Regression on other numeric features.")

        # Step 1: Upload dataset
        uploaded_file_mvp = st.file_uploader("Upload your CSV file", type=["csv"], key="missing_value_uploader")
        
        df_mvp = None # Initialize df_mvp to None
        if uploaded_file_mvp is not None:
            try:
                df_mvp = pd.read_csv(uploaded_file_mvp)
                st.write("### Original Dataset Preview:", df_mvp.head())
            except Exception as e:
                st.error(f"Error loading file: {e}. Please ensure it's a valid CSV.")
                df_mvp = None # Ensure df_mvp is None if loading fails

        if df_mvp is not None: # Proceed only if df_mvp is loaded successfully
            if 'Y' not in df_mvp.columns:
                st.error("❌ Error: The uploaded dataset must contain a column named **'Y'** for prediction.")
            else:
                missing_count = df_mvp['Y'].isnull().sum()
                if missing_count == 0:
                    st.success("✅ Great! There are no missing values in the 'Y' column. Nothing to predict.")
                    st.dataframe(df_mvp)
                else:
                    st.info(f"Detected **{missing_count}** missing values in the 'Y' column. Proceeding with prediction.")

                    df_known = df_mvp[df_mvp['Y'].notnull()].copy()
                    df_missing = df_mvp[df_mvp['Y'].isnull()].copy()

                    # Prepare features (exclude 'Y' and non-numeric columns)
                    numeric_cols_known = df_known.select_dtypes(include=np.number).columns.drop('Y', errors='ignore')
                    numeric_cols_missing = df_missing.select_dtypes(include=np.number).columns.drop('Y', errors='ignore')

                    # Align columns to ensure both dataframes have the same features
                    common_numeric_cols = list(set(numeric_cols_known) & set(numeric_cols_missing))
                    if not common_numeric_cols:
                        st.error("❌ No common numeric features found to train the model. Cannot predict missing 'Y' values.")
                        # This stop is fine here as it's a hard stop if core condition isn't met
                        st.stop() 

                    X_known = df_known[common_numeric_cols]
                    y_known = df_known['Y']
                    X_missing = df_missing[common_numeric_cols]

                    # Handle any remaining NaN in feature columns by filling with mean (or other strategy)
                    X_known = X_known.fillna(X_known.mean())
                    X_missing = X_missing.fillna(X_missing.mean())


                    if X_known.empty or y_known.empty:
                        st.error("Not enough complete data points to train the model. Cannot predict missing 'Y' values.")
                    else:
                        # Step 4: Train Linear Regression model
                        model_mvp = LinearRegression()
                        model_mvp.fit(X_known, y_known)

                        # Step 5: Predict missing Y values
                        y_pred_mvp = model_mvp.predict(X_missing)

                        # Step 6: Replace missing values in original dataset
                        df_mvp.loc[df_mvp['Y'].isnull(), 'Y'] = y_pred_mvp

                        st.write("### Updated Dataset with Missing Y Filled", df_mvp)

                        # Step 7: Allow download of updated dataset
                        csv_output = df_mvp.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download Updated CSV",
                            data=csv_output,
                            file_name="updated_dataset_with_y_filled.csv",
                            mime='text/csv'
                        )

    elif ml_nlp_project == "Sentiment Analysis":
        st.subheader("🤖 Alexa Sentiment Classifier")
        st.markdown("Analyze the sentiment of Alexa product reviews (positive or negative).")

        uploaded_alexa_file = st.file_uploader("📤 Upload 'amazon_alexa.tsv'", type=["tsv", "csv"], key="alexa_uploader_sentiment")

        df_alexa = None
        if uploaded_alexa_file is not None:
            try:
                df_alexa = load_alexa_data_from_path(uploaded_alexa_file)
            except Exception as e:
                st.error(f"Error loading Alexa file: {e}. Please ensure it's a valid TSV/CSV.")
                df_alexa = None
        else: # Try to load from local file if no upload
            if os.path.exists("amazon_alexa.tsv"):
                df_alexa = load_alexa_data_from_path("amazon_alexa.tsv")

        if df_alexa is None:
            st.warning("Please upload 'amazon_alexa.tsv' to use this feature, or place it in the same directory as the script.")
            st.stop() # Stop execution if data isn't loaded

        # --- Preprocess Data ---
        # Renamed to avoid potential global conflicts if not careful
        def clean_text_for_sentiment(text):
            if isinstance(text, str):
                text = text.lower()
                text = re.sub(r'\d+', '', text)  # remove numbers
                text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
                text = text.strip()
            else:
                text = ""
            return text

        if 'verified_reviews' not in df_alexa.columns or 'feedback' not in df_alexa.columns:
            st.error("❌ Dataset must contain 'verified_reviews' and 'feedback' columns.")
            st.stop()

        df_alexa['cleaned'] = df_alexa['verified_reviews'].apply(clean_text_for_sentiment)

        # --- Train Model (Cached) ---
        @st.cache_resource
        def train_sentiment_model_for_alexa(X_data, y_data):
            tfidf = TfidfVectorizer(max_features=5000)
            X_transformed = tfidf.fit_transform(X_data)
            model = LogisticRegression(max_iter=1000)
            model.fit(X_transformed, y_data)
            return model, tfidf

        model_sentiment, vectorizer_sentiment = train_sentiment_model_for_alexa(df_alexa['cleaned'], df_alexa['feedback'])

        # --- User Input & Prediction ---
        review = st.text_area(
            "📨 Enter an Alexa product review",
            "Alexa is very helpful and responsive; I love this device!",
            height=150
        )

        if st.button("🔍 Predict Sentiment", type="primary"):
            if not review.strip():
                st.warning("⚠️ Please enter a review to analyze.")
            else:
                cleaned_review = clean_text_for_sentiment(review)
                vect_input = vectorizer_sentiment.transform([cleaned_review])
                prediction = model_sentiment.predict(vect_input)[0]
                label = "👍 Positive Sentiment" if prediction == 1 else "👎 Negative Sentiment"
                st.success(f"💬 **Predicted Sentiment:** {label}")

        # --- Show Dataset Preview ---
        with st.expander("📊 View Dataset Sample (First 10 Rows)"):
            st.dataframe(df_alexa[['verified_reviews', 'feedback']].head(10), use_container_width=True)

    elif ml_nlp_project == "Titanic Survival Predictor":
        st.subheader("🚢 Titanic Dataset Explorer & Survival Predictor")
        st.markdown("Predict the survival outcome of passengers on the Titanic using a Machine Learning model. Based on `Titanic-Dataset.csv`.")
        
        # Load data via uploader or local file
        uploaded_titanic_file = st.file_uploader("Upload 'Titanic-Dataset.csv'", type=["csv"], key="titanic_uploader_mlnlp")

        df_titanic = None
        if uploaded_titanic_file is not None:
            try:
                df_titanic = load_titanic_data_from_path(uploaded_titanic_file)
            except Exception as e:
                st.error(f"Error loading Titanic file: {e}. Please ensure it's a valid CSV.")
                df_titanic = None
        else: # Try to load from local file if no upload
            if os.path.exists("Titanic-Dataset.csv"):
                df_titanic = load_titanic_data_from_path("Titanic-Dataset.csv")

        if df_titanic is None:
            st.warning("Please upload 'Titanic-Dataset.csv' to use this feature, or place it in the same directory as the script.")
            st.stop() # Stop execution if data isn't loaded
        
        # Show raw data
        if st.checkbox("Show Raw Data"):
            st.write(df_titanic)

        # Clean data (ensure this is done consistently for training and prediction)
        df_clean_titanic = df_titanic.copy()
        df_clean_titanic = df_clean_titanic.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")
        # Fill missing 'Age' with median before encoding, then drop rows with remaining NaNs
        df_clean_titanic['Age'].fillna(df_clean_titanic['Age'].median(), inplace=True)
        df_clean_titanic.dropna(subset=['Embarked', 'Sex'], inplace=True) # Ensure 'Embarked' and 'Sex' have no NaNs

        # Encode categorical features
        # Ensure LabelEncoders are re-initialized and fitted to the full cleaned data
        le_sex_titanic = LabelEncoder()
        le_embarked_titanic = LabelEncoder()
        df_clean_titanic['Sex'] = le_sex_titanic.fit_transform(df_clean_titanic['Sex'])
        df_clean_titanic['Embarked'] = le_embarked_titanic.fit_transform(df_clean_titanic['Embarked'])

        # Input for prediction
        st.markdown("---")
        st.subheader("🎯 Predict Survival for a Passenger")

        col1_pred, col2_pred = st.columns(2)
        with col1_pred:
            pclass_input = st.selectbox("Passenger Class (1=1st, 2=2nd, 3=3rd)", [1, 2, 3], index=2, key="pclass_input_mlnlp")
            sex_input = st.selectbox("Sex", ['male', 'female'], key="sex_input_mlnlp")
            age_input = st.slider("Age", 0, 80, 30, key="age_input_mlnlp")
            sibsp_input = st.slider("Siblings/Spouses Aboard", 0, 5, 0, key="sibsp_input_mlnlp")
        with col2_pred:
            parch_input = st.slider("Parents/Children Aboard", 0, 5, 0, key="parch_input_mlnlp")
            fare_input = st.slider("Fare Paid", 0.0, 500.0, 50.0, key="fare_input_mlnlp")
            
            # Use inverse_transform to get original labels for selectbox, then transform back for model
            unique_embarked_labels = [le_embarked_titanic.inverse_transform([val])[0] for val in sorted(df_clean_titanic['Embarked'].unique())]
            selected_embarked_label = st.selectbox("Port of Embarkation", unique_embarked_labels, key="embarked_input_mlnlp")
            embarked_encoded_user = le_embarked_titanic.transform([selected_embarked_label])[0]

        # Preprocess user input
        sex_encoded_user = le_sex_titanic.transform([sex_input])[0]

        user_data_titanic = pd.DataFrame({
            'Pclass': [pclass_input],
            'Sex': [sex_encoded_user],
            'Age': [age_input],
            'SibSp': [sibsp_input],
            'Parch': [parch_input],
            'Fare': [fare_input],
            'Embarked': [embarked_encoded_user]
        })

        # Train model (cached for performance)
        @st.cache_resource
        def train_titanic_model_final(df_cleaned_input):
            X = df_cleaned_input.drop('Survived', axis=1)
            y = df_cleaned_input['Survived']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            return model, X_test, y_test

        model_titanic, X_test_titanic, y_test_titanic = train_titanic_model_final(df_clean_titanic)

        # Accuracy
        acc = accuracy_score(y_test_titanic, model_titanic.predict(X_test_titanic))
        st.markdown(f"**Model Accuracy on Test Data:** {acc:.2%}")

        # Prediction
        if st.button("Predict Survival for Passenger"):
            prediction = model_titanic.predict(user_data_titanic)[0]
            result = "🟢 **Survived!**" if prediction == 1 else "🔴 **Did Not Survive**"
            st.subheader("Prediction Result")
            if prediction == 1:
                st.success(result)
            else:
                st.error(result)

        # Visualizations
        st.subheader("📊 Data Visualizations")

        col1_vis, col2_vis = st.columns(2)

        with col1_vis:
            st.markdown("#### Survival Count by Sex")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.countplot(data=df_titanic, x='Survived', hue='Sex', ax=ax1, palette='viridis')
            ax1.set_xticks([0, 1])
            ax1.set_xticklabels(['Died', 'Survived'])
            ax1.set_title('Survival Count by Sex')
            st.pyplot(fig1)

        with col2_vis:
            st.markdown("#### Age Distribution")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.histplot(data=df_titanic, x='Age', bins=30, kde=True, ax=ax2, palette='coolwarm')
            ax2.set_title('Age Distribution of Passengers')
            st.pyplot(fig2)

        st.markdown("#### Passenger Class Distribution by Survival")
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        sns.countplot(data=df_titanic, x='Pclass', hue='Survived', ax=ax3, palette='plasma')
        ax3.set_xticks([0, 1, 2])
        ax3.set_xticklabels(['1st Class', '2nd Class', '3rd Class'])
        ax3.set_title('Survival Rate by Passenger Class')
        st.pyplot(fig3)

        st.markdown("---")
        st.markdown("🔧 Built with Streamlit & Scikit-learn | Dataset: Titanic CSV")

# ------------------ DEVOPS PROJECTS ------------------
with tabs[8]:  # This is the 9th tab (index 8)
    st.header("🚀 DevOps Projects")
    st.markdown("""
    Explore my DevOps projects with direct links to GitHub repositories. 
    Each project demonstrates different aspects of modern DevOps practices.
    """)
    
    # Create columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("CI/CD Pipeline")
        st.image("https://cdn-icons-png.flaticon.com/512/270/270798.png", width=100)
        st.markdown("""
        A complete CI/CD pipeline implementation with:
        - Jenkins automation
        - Docker integration
        - Automated testing
        - Deployment strategies
        """)
        if st.button("View CI/CD Project", key="ci_cd_button"):
            st.markdown("[GitHub Repository](https://github.com/rudrapratap0001/ci-cd)", unsafe_allow_html=True)
            st.components.v1.html(
                """
                <script>
                window.open("https://github.com/rudrapratap0001/ci-cd", "_blank");
                </script>
                """,
                height=0
            )
    
    with col2:
        st.subheader("Microservices")
        st.image("https://cdn-icons-png.flaticon.com/512/6132/6132222.png", width=100)
        st.markdown("""
        Microservices architecture demonstrating:
        - Service decomposition
        - API gateways
        - Inter-service communication
        - Container orchestration
        """)
        if st.button("View Microservices Project", key="microservices_button"):
            st.markdown("[GitHub Repository](https://github.com/rudrapratap0001/micro_services)", unsafe_allow_html=True)
            st.components.v1.html(
                """
                <script>
                window.open("https://github.com/rudrapratap0001/micro_services", "_blank");
                </script>
                """,
                height=0
            )
    
    with col3:
        st.subheader("Kubernetes")
        st.image("https://cdn-icons-png.flaticon.com/512/6125/6125000.png", width=100)
        st.markdown("""
        Kubernetes implementations featuring:
        - Cluster deployments
        - Pod management
        - Service discovery
        - Scaling strategies
        """)
        if st.button("View Kubernetes Project", key="kubernetes_button"):
            st.markdown("[GitHub Repository](https://github.com/rudrapratap0001/kubernetes)", unsafe_allow_html=True)
            st.components.v1.html(
                """
                <script>
                window.open("https://github.com/rudrapratap0001/kubernetes", "_blank");
                </script>
                """,
                height=0
            )
    
    st.markdown("---")
    st.markdown("""
    ### Project Highlights
    - All projects include detailed documentation
    - Ready-to-use configuration files
    - Step-by-step setup guides
    - Best practices implementation
    """)