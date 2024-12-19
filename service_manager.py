import subprocess
import time
import logging
import os
import signal
import sys
from logging.handlers import RotatingFileHandler

# 配置日志
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    handlers=[RotatingFileHandler('logs/service_manager.log', maxBytes=1024*1024, backupCount=5)],
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('service_manager')

class ServiceManager:
    def __init__(self):
        self.processes = {}
        self.running = True
        
    def start_service(self, name, command):
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            self.processes[name] = process
            logger.info(f"Started {name} with PID {process.pid}")
            return True
        except Exception as e:
            logger.error(f"Error starting {name}: {str(e)}")
            return False

    def check_and_restart(self):
        for name, process in self.processes.items():
            if process.poll() is not None:  # 进程已结束
                logger.warning(f"{name} is down, restarting...")
                if name == "main":
                    self.start_service("main", ["python", "main.py"])
                elif name == "api":
                    self.start_service("api", ["python", "api_service.py"])

    def stop_all(self):
        self.running = False
        for name, process in self.processes.items():
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            logger.info(f"Stopped {name}")

    def handle_signal(self, signum, frame):
        logger.info("Received shutdown signal, stopping all services...")
        self.stop_all()
        sys.exit(0)

    def run(self):
        # 注册信号处理
        signal.signal(signal.SIGTERM, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)

        # 启动服务
        self.start_service("main", ["python", "main.py"])
        self.start_service("api", ["python", "api_service.py"])

        # 监控循环
        while self.running:
            self.check_and_restart()
            time.sleep(60)  # 每分钟检查一次

if __name__ == "__main__":
    manager = ServiceManager()
    try:
        manager.run()
    except KeyboardInterrupt:
        manager.stop_all() 