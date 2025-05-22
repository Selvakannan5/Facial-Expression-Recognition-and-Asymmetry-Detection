from datetime import datetime

def log_alert(expression, asymmetry):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alert_msg = f"[{timestamp}] Alert: {expression} | Asymmetry: {asymmetry}"
    print(alert_msg)
    with open("alerts_log.txt", "a") as f:
        f.write(alert_msg + "\n")