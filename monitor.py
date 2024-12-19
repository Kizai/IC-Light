import requests
import time
import subprocess
import logging
from logging.handlers import RotatingFileHandler

# 配置日志
logging.basicConfig(
    handlers=[RotatingFileHandler('logs/monitor.log', maxBytes=1024*1024, backupCount=5)],
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('monitor')

def check_service(port, service_name):
    try:
        response = requests.get(f'http://localhost:{port}/health', timeout=5)
        if response.status_code == 200:
            return True
        logger.warning(f"{service_name} health check failed with status code: {response.status_code}")
        return False
    except Exception as e:
        logger.error(f"Error checking {service_name}: {str(e)}")
        return False

def restart_service(service_name):
    try:
        subprocess.run(['supervisorctl', 'restart', f'ic-light:{service_name}'], check=True)
        logger.info(f"Restarted {service_name}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error restarting {service_name}: {str(e)}")

def main():
    while True:
        # 检查主服务
        if not check_service(7860, 'ic-light-main'):
            restart_service('ic-light-main')
        
        # 检查API服务
        if not check_service(8000, 'ic-light-api'):
            restart_service('ic-light-api')
        
        # 每分钟检查一次
        time.sleep(60)

if __name__ == "__main__":
    logger.info("Monitor service started")
    main() 